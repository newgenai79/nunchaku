#pragma once

#include "common.h"
#include "Tensor.h"
#include "debug.h"

class Module {
protected:
    enum class ParamFlags : int {
        None = 0,
        Optional = 1,
    };
    struct Param {
        Tensor *tensor;
        ParamFlags flags;
    };

    friend inline ParamFlags operator|(ParamFlags lhs, ParamFlags rhs) {
        return static_cast<ParamFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
    }
    friend inline ParamFlags operator&(ParamFlags lhs, ParamFlags rhs) {
        return static_cast<ParamFlags>(static_cast<int>(lhs) & static_cast<int>(rhs));
    }

public:
    std::string getFullName() const {
        if (!parent) {
            return name;
        }
        std::string fullName = parent->getFullName();
        if (fullName.empty()) {
            return name;
        } else {
            return fullName + "." + name;
        }
    }

    void traverse(std::function<void(Module *)> func) {
        func(this);
        for (Module *c : this->children) {
            c->traverse(func);
        }
    }

    virtual void loadParams(TensorsProvider &provider, bool partial = false) {
        for (Module *c : children) {
            c->loadParams(provider, partial);
        }
        std::string fullName = getFullName();
        std::string prefix = fullName.empty() ? "" : fullName + ".";
        for (auto &&[key, param] : params) {
            Tensor src = provider.getTensor(prefix + key);
            if (!src.valid()) {
                if (partial || int(param.flags & ParamFlags::Optional)) {
                    continue;
                }
                throw std::runtime_error(spdlog::fmt_lib::format("Tensor {} not found", prefix + key));
            }
            this->loadParam(key, *param.tensor, src);
            // tensor->copy_(src);
        }
    }

    void setName(std::string name) {
        assert(!parent);
        this->name = std::move(name);
    }



protected:
    virtual void loadParam(std::string key, Tensor &dst, Tensor src) {
        dst.copy_(src);
    }

    struct ChildrenRegisterHelper {
        ChildrenRegisterHelper(Module &self) : self(self) {}
        Module &self;
        ChildrenRegisterHelper operator()(Module &module, std::string name) {
            return self.registerChildren(module, name);
        }
    };
    ChildrenRegisterHelper registerChildren(Module &module, std::string name) {
        module.parent = this;
        module.name = name;
        children.push_back(&module);
        return ChildrenRegisterHelper(*this);
    }

    struct ParamsRegisterHelper {
        ParamsRegisterHelper(Module &self) : self(self) {}
        Module &self;
        ParamsRegisterHelper operator()(Tensor &param, std::string name, ParamFlags flags = ParamFlags::None) {
            return self.registerParams(param, name, flags);
        }
    };
    ParamsRegisterHelper registerParams(Tensor &param, std::string name, ParamFlags flags = ParamFlags::None) {
        if (param.valid()) {
            params[name].tensor = &param;
            params[name].flags = flags;
        }
        return ParamsRegisterHelper(*this);
    }

    void debug(std::string name, Tensor tensor) {
        if (DebugContext::ctxs.empty()) {
            return;
        }
        std::string prefix = getFullName();
        if (!prefix.empty()) {
            prefix += ".";
        }
        tensor = tensor.copy(Device::cpu());
        for (auto &&ctx : DebugContext::ctxs) {
            ctx->tensors[prefix + name] = tensor;
        }
    }

public:
    Module *parent = nullptr;
    std::string name = "";
    std::vector<Module *> children;
    std::map<std::string, Param> params;
};
