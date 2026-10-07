#include <chrono>
#include <cstdio>
#include <queue>
#include <vector>
#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

// ---------------- I/O helpers ----------------
std::vector<std::string> BpeTokenizer::LoadVocab(const std::string& vocab_path) {
    std::ifstream ifs(vocab_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open vocab file: " + vocab_path);
    }
    std::vector<std::string> vocab;
    vocab.reserve(260000);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "\\n") {
            vocab.push_back("\n");
        } else if (line == "\\r") {
            vocab.push_back("\r");
        } else {
            vocab.push_back(line);
        }
    }
    return vocab;
}

std::unordered_map<std::string, int> BpeTokenizer::BuildTokenToId(const std::vector<std::string>& id_to_token) {
    std::unordered_map<std::string, int> m;
    m.reserve(id_to_token.size() * 2);
    for (size_t i = 0; i < id_to_token.size(); ++i) {
        m.emplace(id_to_token[i], static_cast<int>(i));
    }
    return m;
}

std::unordered_map<std::string, int> BpeTokenizer::LoadMergesRank(const std::string& merges_path) {
    std::ifstream ifs(merges_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open merges file: " + merges_path);
    }
    std::unordered_map<std::string, int> merges;
    merges.reserve(50000);
    std::string line;
    int rank = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        merges.emplace(PairKey(a, b), rank++);
    }
    return merges;
}

// ---------------- UTF-8 utilities ----------------

bool BpeTokenizer::IsAsciiSpace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool BpeTokenizer::IsUnicodeSpace(uint32_t cp) {
    if (cp <= 0x7F) return IsAsciiSpace(static_cast<unsigned char>(cp));
    switch (cp) {
        case 0x00A0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003:
        case 0x2004: case 0x2005: case 0x2006: case 0x2007:
        case 0x2008: case 0x2009: case 0x200A:
        case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return false;
    }
}

bool BpeTokenizer::NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len) {
    if (i >= s.size()) return false;
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0; cp_len = 1; ++i; return true;
    } else if ((c0 >> 5) == 0x6) {
        if (i + 1 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        if ((c1 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        cp_len = 2; i += 2; return true;
    } else if ((c0 >> 4) == 0xE) {
        if (i + 2 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        unsigned char c2 = static_cast<unsigned char>(s[i+2]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x0F) << 12) |
             ((c1 & 0x3F) << 6) |
             (c2 & 0x3F);
        cp_len = 3; i += 3; return true;
    } else if ((c0 >> 3) == 0x1E) {
        if (i + 3 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        unsigned char c2 = static_cast<unsigned char>(s[i+2]);
        unsigned char c3 = static_cast<unsigned char>(s[i+3]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x07) << 18) |
             ((c1 & 0x3F) << 12) |
             ((c2 & 0x3F) << 6) |
             (c3 & 0x3F);
        cp_len = 4; i += 4; return true;
    }
    return false;
}

std::vector<std::string> BpeTokenizer::Utf8Chars(const std::string& s) {
    std::vector<std::string> chars;
    chars.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp; size_t cp_len;
        if (!NextUtf8(s, i, cp, cp_len)) break;
        chars.emplace_back(s.substr(start, i - start));
    }
    return chars;
}

// ---------------- Byte Encoder/Decoder (New Logic) ----------------

void BpeTokenizer::InitByteMaps() {
    byte_encoder_.resize(256, 0);
    byte_decoder_.clear();

    // Logic based on GPT-2 byte encoder (map printable chars to themselves, others to 256+)
    auto is_printable = [](int b) {
        return (b >= '!' && b <= '~')     // '!' to '~'
                || (b >= 161 && b <= 172)  // '¡' to '¬'
                || (b >= 174 && b <= 255); // '®' to 'ÿ'
    };

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(b)) {
            byte_encoder_[b] = static_cast<uint32_t>(b);
        } else {
            byte_encoder_[b] = static_cast<uint32_t>(256 + n);
            n++;
        }
        // Build decoder map
        byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
    }
}

std::string BpeTokenizer::ByteEncode(const std::string& text) const {
    std::string out;
    out.reserve(text.size() * 2);

    for (unsigned char b : text) {
        uint32_t cp = byte_encoder_[b];
        // Append cp as UTF-8
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>((cp >> 6) | 0xC0);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else if (cp < 0x10000) {
            out += static_cast<char>((cp >> 12) | 0xE0);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else {
            out += static_cast<char>((cp >> 18) | 0xF0);
            out += static_cast<char>(((cp >> 12) & 0x3F) | 0x80);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
    }
    return out;
}

std::string BpeTokenizer::ByteDecode(const std::string& text) const {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    uint32_t cp; size_t len;
    while (i < text.size()) {
        if (!NextUtf8(text, i, cp, len)) break;

        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) {
            out += static_cast<char>(it->second);
        } else {
            // If codepoint is not in byte map, usually we ignore it or keep it
            // if it's part of a special token that wasn't filtered.
            // Here we ignore unmapped chars to ensure pure byte restoration.
        }
    }
    return out;
}

// ---------------- Pretokenizer ----------------
std::vector<std::string> BpeTokenizer::PretokenizeSentencePiece(const std::string& text) {
    static const std::string ws_mark = "▁";
    std::vector<std::string> out;
    std::string curr;
    curr.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp; size_t len;
        if (!NextUtf8(text, i, cp, len)) break;
        if (IsUnicodeSpace(cp)) {
            if (!curr.empty()) {
                out.push_back(ws_mark + curr);
                curr.clear();
            }
        } else {
            curr.append(text, i - len, len);
        }
    }
    if (!curr.empty()) {
        out.push_back(ws_mark + curr);
    }
    return out;
}

// ---------------- BPE core ----------------

std::string BpeTokenizer::PairKey(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);
    key.push_back('\t');
    key.append(b);
    return key;
}

const std::vector<std::string>& BpeTokenizer::BpeForPieceCached(const std::string& piece) const {
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto it = bpe_cache_.find(piece);
        if (it != bpe_cache_.end()) return it->second;
    }
    auto tokens = BpeForPiece(piece);
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto [it, _] = bpe_cache_.emplace(piece, std::move(tokens));
        return it->second;
    }
}

std::vector<std::string> BpeTokenizer::BpeForPiece(const std::string& piece) const {
    std::vector<std::string> symbols = Utf8Chars(piece);
    const int n0 = static_cast<int>(symbols.size());
    if (n0 <= 1) return symbols;

    /* 原实现是 O(n^2 * L)：每轮都重新扫描全部相邻对并做哈希查找，再用 vector::erase
     * 做 O(n) 搬移。实测 2038 字符的整段 prompt 走一次要 470 ms（占 tokenizer 的 99.6%）。
     * 这里改成标准做法：双向链表维护符号序列 + 小顶堆取最小 rank 的相邻对。
     * 复杂度 O(n log n)，且【保持原有的语义】：原实现用 `r < best_rank` 严格小于，
     * 因此同 rank 时取最左的那一对 —— 堆的比较器用 (rank, left) 排序即可复现该行为。 */
    std::vector<int> prev(n0), next(n0);
    std::vector<char> alive(n0, 1);
    for (int i = 0; i < n0; i++) { prev[i] = i - 1; next[i] = i + 1; }
    next[n0 - 1] = -1;

    struct PairNode { int rank; int left; };
    struct PairCmp {
        bool operator()(const PairNode& a, const PairNode& b) const {
            if (a.rank != b.rank) return a.rank > b.rank;  // 小顶堆
            return a.left > b.left;                        // 同 rank 取最左 ✓
        }
    };
    std::priority_queue<PairNode, std::vector<PairNode>, PairCmp> pq;

    auto push_pair = [&](int left) {
        if (left < 0 || !alive[left]) return;
        const int right = next[left];
        if (right < 0 || !alive[right]) return;
        auto it = merges_rank_.find(PairKey(symbols[left], symbols[right]));
        if (it != merges_rank_.end()) pq.push(PairNode{it->second, left});
    };

    for (int i = 0; i + 1 < n0; i++) push_pair(i);

    while (!pq.empty()) {
        const PairNode node = pq.top();
        pq.pop();
        const int l = node.left;
        if (!alive[l]) continue;
        const int r = next[l];
        if (r < 0 || !alive[r]) continue;
        /* 惰性校验：该对的 rank 可能已因合并而改变 */
        auto it = merges_rank_.find(PairKey(symbols[l], symbols[r]));
        if (it == merges_rank_.end() || it->second != node.rank) continue;

        symbols[l] += symbols[r];   /* 合并到左符号 ✓ */
        alive[r] = 0;
        next[l] = next[r];
        if (next[r] >= 0) prev[next[r]] = l;
        push_pair(l);               /* 新形成的邻接对 ✓ */
        if (prev[l] >= 0) push_pair(prev[l]);
    }

    std::vector<std::string> out;
    out.reserve(n0);
    for (int i = 0; i >= 0; i = next[i]) out.push_back(std::move(symbols[i]));
    return out;
}

void BpeTokenizer::TokensToIds(const std::vector<std::string>& tokens, std::vector<int>& out) const {
    for (const auto& t : tokens) {
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (fallback_to_chars_) {
            auto chars = Utf8Chars(t);
            for (const auto& ch : chars) {
                auto it2 = token_to_id_.find(ch);
                if (it2 != token_to_id_.end()) {
                    out.push_back(it2->second);
                } else if (special_ids_.unk_id >= 0) {
                    out.push_back(special_ids_.unk_id);
                }
            }
        } else if (special_ids_.unk_id >= 0) {
            out.push_back(special_ids_.unk_id);
        }
    }
}

// ---------------- Move semantics ----------------
BpeTokenizer::BpeTokenizer(BpeTokenizer&& other) noexcept
    : id_to_token_(std::move(other.id_to_token_)),
      token_to_id_(std::move(other.token_to_id_)),
      merges_rank_(std::move(other.merges_rank_)),
      special_ids_(other.special_ids_),
      fallback_to_chars_(other.fallback_to_chars_),
      additional_special_tokens_(std::move(other.additional_special_tokens_)),
      additional_special_token_ids_(std::move(other.additional_special_token_ids_)),
      additional_special_token_to_id_(std::move(other.additional_special_token_to_id_)),
      additional_special_id_set_(std::move(other.additional_special_id_set_)),
      use_byte_encoder_(other.use_byte_encoder_),
      byte_encoder_(std::move(other.byte_encoder_)),
      byte_decoder_(std::move(other.byte_decoder_)) {
    // 迁移缓存（为安全起见加锁）
    std::lock_guard<std::mutex> lk(other.cache_mu_);
    bpe_cache_ = std::move(other.bpe_cache_);
    // cache_mu_ 自身是新构造的 mutex
}

BpeTokenizer& BpeTokenizer::operator=(BpeTokenizer&& other) noexcept {
    if (this != &other) {
        // 同时锁两边，避免数据竞争
        std::scoped_lock lk(cache_mu_, other.cache_mu_);
        id_to_token_ = std::move(other.id_to_token_);
        token_to_id_ = std::move(other.token_to_id_);
        merges_rank_ = std::move(other.merges_rank_);
        special_ids_ = other.special_ids_;
        fallback_to_chars_ = other.fallback_to_chars_;
        additional_special_tokens_ = std::move(other.additional_special_tokens_);
        additional_special_token_ids_ = std::move(other.additional_special_token_ids_);
        additional_special_token_to_id_ = std::move(other.additional_special_token_to_id_);
        additional_special_id_set_ = std::move(other.additional_special_id_set_);
        use_byte_encoder_ = other.use_byte_encoder_;
        byte_encoder_ = std::move(other.byte_encoder_);
        byte_decoder_ = std::move(other.byte_decoder_);
        bpe_cache_ = std::move(other.bpe_cache_);
        // cache_mu_ 仍为当前对象自己的 mutex
    }
    return *this;
}

// ---------------- Public APIs ----------------
BpeTokenizer BpeTokenizer::LoadFromFiles(const std::string& vocab_path,
                                         const std::string& merges_path,
                                         const SpecialTokensConfig& spec,
                                         bool add_special_if_missing,
                                         bool fallback_to_chars,
                                         bool use_byte_encoder) {
    BpeTokenizer tok;
    tok.id_to_token_ = LoadVocab(vocab_path);
    if (tok.id_to_token_.empty()) {
        throw std::runtime_error("Vocab is empty: " + vocab_path);
    }
    tok.token_to_id_ = BuildTokenToId(tok.id_to_token_);
    tok.merges_rank_ = LoadMergesRank(merges_path);
    tok.fallback_to_chars_ = fallback_to_chars;
    tok.use_byte_encoder_ = use_byte_encoder;

    if (tok.use_byte_encoder_) {
        tok.InitByteMaps();
    }

    tok.EnsureSpecialTokens(spec, add_special_if_missing);
    // 显式移动，避免 MSVC 尝试拷贝
    return std::move(tok);
}

void BpeTokenizer::EnsureSpecialTokens(const SpecialTokensConfig& spec, bool add_if_missing) {
    auto ensure = [&](const std::optional<std::string>& name, int& id_slot) {
        if (!name.has_value()) { id_slot = -1; return; }
        auto it = token_to_id_.find(*name);
        if (it != token_to_id_.end()) {
            id_slot = it->second;
            return;
        }
        if (!add_if_missing) {
            id_slot = -1;
            return;
        }
        id_slot = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(*name);
        token_to_id_.emplace(*name, id_slot);
    };

    ensure(spec.bos_token, special_ids_.bos_id);
    ensure(spec.eos_token, special_ids_.eos_id);
    ensure(spec.unk_token, special_ids_.unk_id);
    ensure(spec.sep_token, special_ids_.sep_id);
    ensure(spec.pad_token, special_ids_.pad_id);
    ensure(spec.cls_token, special_ids_.cls_id);
    ensure(spec.mask_token, special_ids_.mask_id);
}

void BpeTokenizer::AddAdditionalSpecialToken(const std::string& token,
                                             bool add_if_missing) {
    if (token.empty()) return;

    // 已经是 additional_special_token 的话直接返回
    auto it_exist = additional_special_token_to_id_.find(token);
    if (it_exist != additional_special_token_to_id_.end()) {
        return;
    }

    int id = -1;
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        id = it->second;
    } else if (add_if_missing) {
        id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_.emplace(token, id);
    } else {
        // 既不在 vocab 中，也不允许新增，直接忽略
        return;
    }

    additional_special_tokens_.push_back(token);
    additional_special_token_ids_.push_back(id);
    additional_special_token_to_id_.emplace(token, id);
    additional_special_id_set_.insert(id);
}

void BpeTokenizer::SetAdditionalSpecialTokens(const std::vector<std::string>& tokens,
                                              bool add_if_missing) {
    additional_special_tokens_.clear();
    additional_special_token_ids_.clear();
    additional_special_token_to_id_.clear();
    additional_special_id_set_.clear();

    for (const auto& t : tokens) {
        AddAdditionalSpecialToken(t, add_if_missing);
    }
}

std::vector<int> BpeTokenizer::encode(const std::string& text,
                                      bool add_bos,
                                      bool add_eos,
                                      bool add_cls,
                                      bool add_sep) const {
    /* 剖析（NCNN_BPE_PROF=1 ✓）：定位 tokenizer 热点 ✓ */
    static const bool bprof = (getenv("NCNN_BPE_PROF") != nullptr);
    using bpe_clock = std::chrono::steady_clock;
    bpe_clock::time_point _bt0;
    if (bprof) _bt0 = bpe_clock::now();
    double _t_scan = 0;

    std::vector<int> ids;
    ids.reserve(text.size() / 2 + 8);

    if (add_cls && special_ids_.cls_id >= 0) ids.push_back(special_ids_.cls_id);
    if (add_bos && special_ids_.bos_id >= 0) ids.push_back(special_ids_.bos_id);

    // 先在原始串中按子串匹配 additional_special_tokens；
    // 命中时：
    //   1) flush 之前累积的普通文本（走 SentencePiece+BPE）
    //   2) 直接输出 special id
    std::string buffer;
    buffer.reserve(text.size());

    auto flush_buffer = [&]() {
        if (buffer.empty()) return;

        if (use_byte_encoder_) {
            // Mode 2: Byte-Level Encoding
            // Encode raw bytes to mapped unicode chars
            std::string encoded_s = ByteEncode(buffer);
            // Perform BPE on the entire encoded string
            const auto& toks = BpeForPieceCached(encoded_s);
            TokensToIds(toks, ids);
        } else {
            auto pieces = PretokenizeSentencePiece(buffer);
            for (const auto& p : pieces) {
                const auto& toks = BpeForPieceCached(p);
                TokensToIds(toks, ids);
            }
        }
        buffer.clear();
    };

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        int matched_index = -1;
        size_t matched_len = 0;

        // longest match：若多个 special token 共享前缀，选择更长的那个
        bpe_clock::time_point _sc0;
        if (bprof) _sc0 = bpe_clock::now();
        if (!additional_special_tokens_.empty()) {
            for (size_t k = 0; k < additional_special_tokens_.size(); ++k) {
                const std::string& sp = additional_special_tokens_[k];
                if (sp.empty()) continue;
                size_t len = sp.size();
                if (len <= matched_len) continue;
                if (i + len <= n && text.compare(i, len, sp) == 0) {
                    matched_index = static_cast<int>(k);
                    matched_len = len;
                }
            }
        }

        if (bprof) {
            auto _sc1 = bpe_clock::now();
            _t_scan += std::chrono::duration<double, std::milli>(_sc1 - _sc0).count();
        }
        if (matched_index >= 0) {
            flush_buffer();
            ids.push_back(additional_special_token_ids_[matched_index]);
            i += matched_len;
            continue;
        }

        buffer.push_back(text[i]);
        ++i;
    }

    flush_buffer();

    if (add_sep && special_ids_.sep_id >= 0) ids.push_back(special_ids_.sep_id);
    if (add_eos && special_ids_.eos_id >= 0) ids.push_back(special_ids_.eos_id);
    if (bprof) {
        auto _bt1 = bpe_clock::now();
        const double tot = std::chrono::duration<double, std::milli>(_bt1 - _bt0).count();
        fprintf(stderr, "[BPE-PROF] text=%zu 字符 special_tokens=%zu → ids=%zu | 总 %.2f ms | 扫描 %.2f ms (%.1f%%) | 其余(BPE等) %.2f ms (%.1f%%)\n",
                text.size(), additional_special_tokens_.size(), ids.size(),
                tot, _t_scan, tot > 0 ? 100.0 * _t_scan / tot : 0.0,
                tot - _t_scan, tot > 0 ? 100.0 * (tot - _t_scan) / tot : 0.0);
        fflush(stderr);
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string s;
    s.reserve(ids.size() * 3);
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];

        bool is_special =
            id == special_ids_.bos_id || id == special_ids_.eos_id ||
            id == special_ids_.unk_id || id == special_ids_.sep_id ||
            id == special_ids_.pad_id || id == special_ids_.cls_id ||
            id == special_ids_.mask_id ||
            (additional_special_id_set_.find(id) != additional_special_id_set_.end());

        if (skip_special_tokens && is_special) continue;

        if (tok == "\\t")
            s += '\t';
        else if (tok == "\\n")
            s += '\n';
        else if (tok == "\\r")
            s += '\r';
        else
            s += tok;
    }

    if (!s.empty()) {
        // Branch based on encoding mode
        if (use_byte_encoder_) {
            return ByteDecode(s);
        } else {
            std::string out;
            out.reserve(s.size());
            size_t i = 0;
            while (i < s.size()) {
                uint32_t cp; size_t len;
                if (!NextUtf8(s, i, cp, len)) break;
                if (cp == 0x2581) {
                    out.push_back(' ');
                } else {
                    out.append(s, i - len, len);
                }
            }
            size_t b = 0;
            while (b < out.size() && out[b] == ' ') ++b;
            if (b > 0) out.erase(0, b);
            return out;
        }
    }
    return s;
}
