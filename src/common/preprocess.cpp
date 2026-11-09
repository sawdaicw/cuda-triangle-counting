#include "preprocess.hpp"
#include <omp.h>
#include <charconv>
#include <exception>
#include <iterator>
#include <numeric>

namespace tc {
namespace {
using Edge = std::pair<std::int64_t, std::int64_t>;
using Clock = std::chrono::steady_clock;
std::size_t cut(std::size_t n, std::size_t i, std::size_t p) { return (n / p) * i + std::min(i, n % p); }

template<class F> void parallel_for(std::size_t n, int threads, int& actual, const F& function) {
    if (n > static_cast<std::size_t>(std::numeric_limits<long long>::max())) throw std::runtime_error("Too many work items");
    if (threads == 1) { actual = std::max(actual, 1); for (std::size_t i = 0; i < n; ++i) function(i); return; }
    int team = 1;
    #pragma omp parallel num_threads(threads) shared(team)
    {
        #pragma omp single
        team = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) function(static_cast<std::size_t>(i));
    }
    actual = std::max(actual, team);
}

// Stable co-ranking: split two sorted ranges at output rank k. Equal values
// from A precede equal values from B, including across parallel merge tiles.
template<class T> std::size_t corank(std::size_t k, const T* a, std::size_t na, const T* b, std::size_t nb) {
    std::size_t lo = k > nb ? k - nb : 0, hi = std::min(k, na);
    while (lo <= hi) {
        const auto i = lo + (hi - lo) / 2, j = k - i;
        if (i && j < nb && b[j] < a[i - 1]) hi = i - 1;
        else if (j && i < na && !(b[j - 1] < a[i])) lo = i + 1;
        else return i;
    }
    throw std::logic_error("Invalid merge partition");
}
template<class T> void sort_unique(std::vector<T>& values, int threads, int& actual) {
    const auto n = values.size();
    if (threads == 1 || n < 2) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end()); return;
    }
    const auto p = std::min<std::size_t>(threads, n);
    struct Range { std::size_t lo, hi; };
    std::vector<Range> runs;
    for (std::size_t i = 0; i < p; ++i) runs.push_back({cut(n, i, p), cut(n, i + 1, p)});
    parallel_for(p, threads, actual, [&](auto i) { std::sort(values.begin() + runs[i].lo, values.begin() + runs[i].hi); });
    std::vector<T> scratch(n);
    const auto tile = (n + p - 1) / p;
    while (runs.size() > 1) {
        struct Job { std::size_t lo, mid, hi, k0, k1; };
        std::vector<Job> jobs;
        std::vector<Range> next;
        for (std::size_t r = 0; r < runs.size(); r += 2) {
            const auto lo = runs[r].lo, mid = runs[r].hi;
            const auto hi = r + 1 < runs.size() ? runs[r + 1].hi : mid;
            next.push_back({lo, hi});
            for (std::size_t k = 0; k < hi - lo; k += tile)
                jobs.push_back({lo, mid, hi, k, std::min(k + tile, hi - lo)});
        }
        parallel_for(jobs.size(), threads, actual, [&](auto index) {
            const auto j = jobs[index];
            const auto* a = values.data() + j.lo;
            const auto* b = values.data() + j.mid;
            const auto na = j.mid - j.lo, nb = j.hi - j.mid;
            const auto i0 = corank(j.k0, a, na, b, nb), i1 = corank(j.k1, a, na, b, nb);
            std::merge(a + i0, a + i1, b + (j.k0 - i0), b + (j.k1 - i1), scratch.begin() + j.lo + j.k0);
        });
        values.swap(scratch); runs.swap(next);
    }
    // Parallel unique: compare with the global predecessor, so duplicates that
    // cross a worker boundary are removed too. Counts -> prefix -> disjoint copy.
    std::vector<std::size_t> counts(p, 0), positions(p + 1, 0);
    parallel_for(p, threads, actual, [&](auto t) {
        for (auto i = cut(n, t, p); i < cut(n, t + 1, p); ++i)
            if (i == 0 || !(values[i] == values[i - 1])) ++counts[t];
    });
    for (std::size_t t = 0; t < p; ++t) positions[t + 1] = positions[t] + counts[t];
    if (positions.back() == n) return;
    scratch.resize(positions.back());
    parallel_for(p, threads, actual, [&](auto t) {
        auto output = positions[t];
        for (auto i = cut(n, t, p); i < cut(n, t + 1, p); ++i)
            if (i == 0 || !(values[i] == values[i - 1])) scratch[output++] = values[i];
    });
    values.swap(scratch);
}

bool space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; }
bool number(const char*& p, const char* end, std::int64_t& value) {
    while (p != end && space(*p)) ++p;
    if (p == end) return false;
    // from_chars deliberately does not accept '+', while the original stream
    // reader does. Preserve that part of the legacy decimal grammar explicitly.
    if (*p == '+') { ++p; if (p == end || *p < '0' || *p > '9') return false; }
    const auto result = std::from_chars(p, end, value);
    if (result.ec != std::errc{} || result.ptr == p) return false;
    p = result.ptr; return true;
}
struct Raw { std::vector<std::int64_t> ids; std::vector<Edge> edges; };
// Match the original getline/stream behavior, including whole-line comments,
// signed 64-bit bounds, trailing-field rejection, and only-loop vertices.
bool line(const char* p, const char* end, Raw& raw) {
    while (p != end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
    if (p == end || *p == '#' || *p == '%') return true;
    std::int64_t u, v;
    if (!number(p, end, u) || !number(p, end, v)) return false;
    while (p != end && space(*p)) ++p;
    if (p != end) return false;
    raw.ids.push_back(u); raw.ids.push_back(v);
    if (u != v) raw.edges.emplace_back(std::min(u, v), std::max(u, v));
    return true;
}
std::runtime_error input_error(std::size_t number) {
    return std::runtime_error("Line " + std::to_string(number) + ": expected two signed 64-bit integer endpoints (plain text, no header)");
}
Raw legacy_parse(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    Raw raw; std::string text; std::size_t line_number = 0;
    while (std::getline(input, text)) {
        ++line_number;
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || text[first] == '#' || text[first] == '%') continue;
        std::istringstream stream(text); std::int64_t u, v; std::string extra;
        if (!(stream >> u >> v) || (stream >> extra)) throw input_error(line_number);
        raw.ids.push_back(u); raw.ids.push_back(v);
        if (u != v) raw.edges.emplace_back(std::min(u, v), std::max(u, v));
    }
    if (input.bad()) throw std::runtime_error("Error reading input");
    return raw;
}
Raw fast_parse(const std::string& path, int threads, PreprocessReport& report, bool profiling) {
    auto previous = profiling ? Clock::now() : Clock::time_point{};
    const auto mark = [&](double& value) { if (profiling) { const auto now = Clock::now(); value = std::chrono::duration<double, std::milli>(now - previous).count(); previous = now; } };
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    const auto size = input.tellg();
    if (size < 0 || static_cast<unsigned long long>(size) > static_cast<unsigned long long>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("Input is not a supported seekable regular file");
    std::vector<char> buffer(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!buffer.empty() && !input.read(buffer.data(), static_cast<std::streamsize>(buffer.size())))
        throw std::runtime_error("Error reading input (file may have changed)");
    report.input_buffer_bytes = buffer.size(); mark(report.file_read_ms);
    const std::size_t p = static_cast<std::size_t>(threads), n = buffer.size();
    std::vector<std::size_t> cuts(p + 1, n); cuts[0] = 0;
    for (std::size_t t = 1; t < p; ++t) {
        auto point = cut(n, t, p);
        if (point && point < n && buffer[point - 1] != '\n') {
            while (point < n && buffer[point] != '\n') ++point;
            if (point < n) ++point;
        }
        cuts[t] = point;
    }
    mark(report.partition_ms);
    std::vector<Raw> local(p);
    std::vector<std::size_t> errors(p, n);
    std::vector<std::exception_ptr> exceptions(p);
    parallel_for(p, threads, report.parser_threads, [&](auto t) {
        try {
            auto pos = cuts[t];
            while (pos < cuts[t + 1]) {
                auto end = pos;
                while (end < cuts[t + 1] && buffer[end] != '\n') ++end;
                if (!line(buffer.data() + pos, buffer.data() + end, local[t])) { errors[t] = pos; break; }
                pos = end < cuts[t + 1] ? end + 1 : end;
            }
        } catch (...) { exceptions[t] = std::current_exception(); }
    });
    for (const auto& exception : exceptions) if (exception) std::rethrow_exception(exception);
    const auto bad = *std::min_element(errors.begin(), errors.end());
    if (bad != n) throw input_error(1 + static_cast<std::size_t>(std::count(buffer.begin(), buffer.begin() + bad, '\n')));
    mark(report.parse_collect_ms);
    std::vector<std::size_t> id_pos(p + 1, 0), edge_pos(p + 1, 0);
    for (std::size_t t = 0; t < p; ++t) { id_pos[t + 1] = id_pos[t] + local[t].ids.size(); edge_pos[t + 1] = edge_pos[t] + local[t].edges.size(); }
    Raw raw; raw.ids.resize(id_pos.back()); raw.edges.resize(edge_pos.back());
    parallel_for(p, threads, report.parser_threads, [&](auto t) {
        if (!local[t].ids.empty()) std::copy(local[t].ids.begin(), local[t].ids.end(), raw.ids.begin() + id_pos[t]);
        if (!local[t].edges.empty()) std::copy(local[t].edges.begin(), local[t].edges.end(), raw.edges.begin() + edge_pos[t]);
    });
    // Include temporary-buffer release in merge/cleanup time, not outside total.
    local.clear(); std::vector<char>().swap(buffer); input.close();
    mark(report.merge_collect_ms);
    return raw;
}
} // namespace

bool consume_preprocess_option(const std::string& flag, int& i, int argc, char** argv, PreprocessOptions& o) {
    if (flag == "--check-csr") { o.check_csr = true; return true; }
    if (flag != "--parser" && flag != "--mapping" && flag != "--sort" && flag != "--threads") return false;
    if (++i == argc) throw std::runtime_error("Missing value for " + flag);
    const std::string value = argv[i];
    if (flag == "--threads") {
        int threads = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), threads);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || threads < 1 || threads > 256)
            throw std::runtime_error("--threads must be 1..256");
        o.threads = threads;
    } else if (flag == "--parser" && (value == "legacy" || value == "fast" || value == "omp")) o.parser = value;
    else if (flag == "--mapping" && (value == "fused" || value == "serial" || value == "omp")) o.mapping = value;
    else if (flag == "--sort" && (value == "serial" || value == "omp")) o.sorting = value;
    else throw std::runtime_error("Invalid value for " + flag + ": " + value);
    return true;
}
Graph read_graph_configured(const std::string& path, const PreprocessOptions& o, PreprocessProfile* profile, PreprocessReport& report) {
    report = {};
    // Preserve the original implementation as the default and frozen control.
    if (o.parser == "legacy" && o.mapping == "fused" && o.sorting == "serial") return read_graph(path, profile);
    if (profile) *profile = {};
    auto previous = profile ? Clock::now() : Clock::time_point{};
    const auto mark = [&](double PreprocessProfile::*field) { if (profile) { const auto now = Clock::now(); profile->*field = std::chrono::duration<double, std::milli>(now - previous).count(); previous = now; } };
    auto raw = o.parser == "legacy" ? legacy_parse(path) : fast_parse(path, o.parser == "omp" ? o.threads : 1, report, profile != nullptr);
    mark(&PreprocessProfile::read_parse_normalize_ms);
    const int sort_threads = o.sorting == "omp" ? o.threads : 1;
    sort_unique(raw.ids, sort_threads, report.sorting_threads);
    if (raw.ids.size() > std::numeric_limits<Vertex>::max()) throw std::runtime_error("Too many vertices for 32-bit vertex indices");
    mark(&PreprocessProfile::vertex_sort_unique_ms);
    sort_unique(raw.edges, sort_threads, report.sorting_threads);
    mark(&PreprocessProfile::edge_sort_unique_ms);
    std::vector<std::pair<Vertex, Vertex>> dense;
    std::vector<Offset> degree(raw.ids.size(), 0);
    const auto map = [&](const Edge& e) { return std::pair<Vertex, Vertex>{
        static_cast<Vertex>(std::lower_bound(raw.ids.begin(), raw.ids.end(), e.first) - raw.ids.begin()),
        static_cast<Vertex>(std::lower_bound(raw.ids.begin(), raw.ids.end(), e.second) - raw.ids.begin())}; };
    if (o.mapping == "fused") {
        dense.reserve(raw.edges.size());
        for (const auto& edge : raw.edges) { const auto e = map(edge); dense.push_back(e); ++degree[e.first]; ++degree[e.second]; }
    } else {
        dense.resize(raw.edges.size());
        parallel_for(raw.edges.size(), o.mapping == "omp" ? o.threads : 1, report.mapping_threads,
                     [&](auto i) { dense[i] = map(raw.edges[i]); });
        // Deliberately serial: avoid degree atomics and retain a separate
        // single-thread control for the extra pass introduced by splitting.
        for (const auto& edge : dense) { ++degree[edge.first]; ++degree[edge.second]; }
    }
    mark(&PreprocessProfile::remap_degree_ms);
    Graph graph; graph.offsets.assign(raw.ids.size() + 1, 0);
    for (auto& edge : dense) {
        auto& u = edge.first; auto& v = edge.second;
        if (degree[u] > degree[v] || (degree[u] == degree[v] && u > v)) std::swap(u, v);
        ++graph.offsets[static_cast<std::size_t>(u) + 1];
    }
    mark(&PreprocessProfile::orient_row_counts_ms);
    for (std::size_t i = 1; i < graph.offsets.size(); ++i) graph.offsets[i] += graph.offsets[i - 1];
    mark(&PreprocessProfile::csr_scan_ms);
    graph.neighbors.resize(dense.size()); auto cursor = graph.offsets;
    for (const auto& edge : dense) graph.neighbors[cursor[edge.first]++] = edge.second;
    mark(&PreprocessProfile::csr_fill_ms);
    for (std::size_t u = 0; u < raw.ids.size(); ++u)
        std::sort(graph.neighbors.begin() + graph.offsets[u], graph.neighbors.begin() + graph.offsets[u + 1]);
    mark(&PreprocessProfile::neighbor_sort_ms);
    return graph;
}
void check_csr_against_legacy(const std::string& path, const Graph& graph) {
    const auto reference = read_graph(path);
    if (reference.offsets != graph.offsets || reference.neighbors != graph.neighbors)
        throw std::runtime_error("Optimized preprocessing differs from legacy CSR");
}
void write_preprocess_options(std::ostream& out, const PreprocessOptions& o, const PreprocessReport& r, bool profiled) {
    out << "{\"parser\":\"" << o.parser << "\",\"mapping\":\"" << o.mapping << "\",\"sort\":\"" << o.sorting
        << "\",\"threads_requested\":" << o.threads << ",\"parser_threads_actual\":" << r.parser_threads
        << ",\"mapping_threads_actual\":" << r.mapping_threads << ",\"sorting_threads_max_actual\":" << r.sorting_threads
        << ",\"input_buffer_bytes\":" << r.input_buffer_bytes << ",\"csr_verified\":" << (o.check_csr ? "true" : "false");
    if (profiled && o.parser != "legacy")
        out << ",\"parser_detail\":{\"file_read_ms\":" << r.file_read_ms << ",\"partition_ms\":" << r.partition_ms
            << ",\"parse_collect_ms\":" << r.parse_collect_ms << ",\"merge_collect_ms\":" << r.merge_collect_ms << '}';
    out << '}';
}
} // namespace tc
