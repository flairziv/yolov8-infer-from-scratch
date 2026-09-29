// model.cpp —— 解析 model.txt、把权重文件读进内存、检查图的合法性
#include "model.h"

#include <fstream>
#include <sstream>

namespace yi {

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

std::string Tensor::shape_str() const {
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? "x" : "") + std::to_string(shape[i]);
    return s;
}

std::vector<int64_t> Attrs::ints(const std::string& key) const {
    const auto it = kv.find(key);
    YI_CHECK(it != kv.end(), "缺少属性 " << key);
    std::vector<int64_t> r;
    for (double v : it->second) {
        const auto iv = static_cast<int64_t>(v);
        YI_CHECK(static_cast<double>(iv) == v, "属性 " << key << " 不是整数: " << v);
        r.push_back(iv);
    }
    return r;
}

int64_t Attrs::i(const std::string& key) const {
    const auto v = ints(key);
    YI_CHECK(v.size() == 1, "属性 " << key << " 应该是一个整数，实际有 " << v.size() << " 个值");
    return v[0];
}

int Model::find(const std::string& name) const {
    const auto it = index.find(name);
    return it == index.end() ? -1 : it->second;
}

namespace {
std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
}
}  // namespace

Model Model::load(const std::string& dir) {
    const std::string path = dir + "/model.txt";
    std::ifstream f(path);
    YI_CHECK(f, "打不开 " << path);

    Model m;
    std::string weights_file;
    int64_t weights_bytes = -1;
    std::vector<std::string> in_names, out_names;
    struct RawNode {                        // 第一遍先按名字记下来，全部读完再把名字换成下标
        std::string op, name;
        std::vector<std::string> ins, outs;
        Attrs attrs;
    };
    std::vector<RawNode> raw;

    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        std::istringstream ss(line);
        std::string kw;
        if (!(ss >> kw) || kw[0] == '#') continue;          // 空行、注释
        if (kw == "format") {
            std::string name;
            int version = 0;
            ss >> name >> version;
            YI_CHECK(name == "yolov8-infer" && version == 1, path << " 不是本运行时能读的格式: " << line);
        } else if (kw == "weights") {
            ss >> weights_file >> weights_bytes;
        } else if (kw == "input" || kw == "output") {
            std::string name;
            ss >> name;
            (kw == "input" ? in_names : out_names).push_back(name);
        } else if (kw == "tensor") {
            Tensor t;
            std::string shape, kind;
            ss >> t.name >> shape >> kind;
            for (const auto& d : split(shape, ',')) t.shape.push_back(std::stoll(d));
            if (kind == "const") {
                t.is_const = true;
                ss >> t.const_offset;
            } else {
                YI_CHECK(kind == "act", path << " 第 " << lineno << " 行: 张量类型只能是 const 或 act");
            }
            YI_CHECK(m.index.count(t.name) == 0, "张量重名: " << t.name);
            m.index[t.name] = static_cast<int>(m.tensors.size());
            m.tensors.push_back(std::move(t));
        } else if (kw == "node") {
            RawNode n;
            ss >> n.op >> n.name;
            std::string tok;
            while (ss >> tok) {                             // 其余的词都是 key=v1,v2,...
                const auto eq = tok.find('=');
                YI_CHECK(eq != std::string::npos, path << " 第 " << lineno << " 行: 看不懂 " << tok);
                const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
                if (key == "in") {
                    n.ins = split(val, ',');
                } else if (key == "out") {
                    n.outs = split(val, ',');
                } else {
                    for (const auto& v : split(val, ',')) n.attrs.kv[key].push_back(std::stod(v));
                }
            }
            raw.push_back(std::move(n));
        } else {
            YI_CHECK(false, path << " 第 " << lineno << " 行: 不认识的关键字 " << kw);
        }
    }

    // 名字 → 下标。节点可能引用写在它后面的张量声明，所以等全部读完再换
    auto id = [&](const std::string& name) {
        const int i = m.find(name);
        YI_CHECK(i >= 0, "引用了没有声明的张量: " << name);
        return i;
    };
    for (auto& r : raw) {
        Node n;
        n.op = std::move(r.op);
        n.name = std::move(r.name);
        n.attrs = std::move(r.attrs);
        for (const auto& s : r.ins) n.inputs.push_back(id(s));
        for (const auto& s : r.outs) n.outputs.push_back(id(s));
        m.nodes.push_back(std::move(n));
    }
    for (const auto& s : in_names) m.inputs.push_back(id(s));
    for (const auto& s : out_names) m.outputs.push_back(id(s));

    // 权重：整个文件一次读进 64 字节对齐的内存，常量张量直接指进去，不再逐个拷贝
    YI_CHECK(!weights_file.empty(), path << " 里缺少 weights 行");
    m.weights = read_file(dir + "/" + weights_file, weights_bytes);
    for (auto& t : m.tensors) {
        if (!t.is_const) continue;
        YI_CHECK(t.const_offset >= 0 && t.const_offset % static_cast<int64_t>(kAlign) == 0, t.name << " 的偏移没有 64 字节对齐");
        YI_CHECK(t.const_offset + static_cast<int64_t>(t.bytes()) <= weights_bytes, t.name << " 超出了权重文件末尾");
        t.data = reinterpret_cast<float*>(m.weights.as<char>() + t.const_offset);
    }
    m.validate();
    return m;
}

void Model::validate() const {
    // ready[t]：执行到当前节点时，张量 t 是否已经有值（常量和图输入一开始就有）
    std::vector<char> ready(tensors.size(), 0);
    for (size_t i = 0; i < tensors.size(); ++i) ready[i] = tensors[i].is_const;
    for (int i : inputs) {
        YI_CHECK(!tensors[i].is_const, "图输入 " << tensors[i].name << " 不能是常量");
        ready[i] = 1;
    }
    for (const Node& n : nodes) {
        for (int i : n.inputs)
            YI_CHECK(ready[i], "节点 " << n.name << " 的输入 " << tensors[i].name << " 在它之前没有产生（节点表不是拓扑序）");
        for (int o : n.outputs) {
            YI_CHECK(!tensors[o].is_const && !ready[o], "张量 " << tensors[o].name << " 被产生了不止一次，或者往常量里写");
            ready[o] = 1;
        }
    }
    for (size_t i = 0; i < tensors.size(); ++i)
        YI_CHECK(ready[i], "激活 " << tensors[i].name << " 没有任何节点产生它");
    YI_CHECK(!outputs.empty(), "模型没有图输出");
}

}  // namespace yi
