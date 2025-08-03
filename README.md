# 彭永强的技术博客

[![Hugo](https://img.shields.io/badge/Hugo-0.148+-ff4088?style=flat&logo=hugo)](https://gohugo.io/)
[![Blowfish](https://img.shields.io/badge/Theme-Blowfish-blue)](https://blowfish.page/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 专注于Java开发、Kubernetes、容器技术、Go语言和AI技术分享。多思考，为防止小年痴呆！

## 🚀 快速开始

### 环境要求

- [Hugo](https://gohugo.io/installation/) >= 0.140.0 (Extended版本)
- [Git](https://git-scm.com/)
- [Node.js](https://nodejs.org/) >= 14.0.0 (可选，用于NPM脚本)

### 安装和运行

```bash
# 克隆仓库
git clone https://github.com/pengyongqiang666/my-hugo.git
cd my-hugo

# 初始化子模块（主题）
git submodule update --init --recursive

# 安装依赖（可选）
npm install

# 启动开发服务器
npm run dev
# 或者直接使用Hugo
hugo server -D
```

访问 [http://localhost:1313](http://localhost:1313) 查看博客。

## 📝 内容管理

### 创建新文章

```bash
# 使用NPM脚本
npm run new-post 文章标题

# 或直接使用Hugo
hugo new posts/文章标题.md
```

### 文章结构

新文章会使用 `archetypes/posts.md` 模板，包含：
- 标题、日期、标签等元信息
- 显示设置（目录、面包屑等）
- 文章模板结构

### 页面类型

- **首页** (`content/_index.md`) - 博客主页
- **关于** (`content/about.md`) - 个人介绍
- **项目** (`content/projects.md`) - 技术项目展示
- **联系** (`content/contact.md`) - 联系方式
- **文章** (`content/posts/`) - 技术文章

## 🛠️ 开发工具

### NPM脚本

| 命令 | 描述 |
|------|------|
| `npm run dev` | 启动开发服务器 |
| `npm run build` | 生产环境构建 |
| `npm run check` | 完整构建检查 |
| `npm run check-dev` | 检查 + 启动开发服务器 |
| `npm run clean` | 清理构建文件 |
| `npm run deploy` | 部署准备 |

### 构建检查

项目包含自动化检查脚本 `scripts/check-build.sh`，会检查：

- ✅ Hugo版本兼容性
- ✅ 配置文件语法
- ✅ 内容文件完整性
- ✅ 构建测试
- ✅ 生成文件验证
- ✅ SEO和性能建议

## 🎨 主题配置

使用 [Blowfish](https://blowfish.page/) 主题，主要配置文件：

- `config/_default/hugo.toml` - Hugo主配置
- `config/_default/params.toml` - 主题参数
- `config/_default/menus.toml` - 导航菜单
- `config/_default/languages.toml` - 语言配置

### 特色功能

- 🌙 **暗色模式** - 支持主题切换
- 🔍 **全站搜索** - 快速内容查找
- 📱 **响应式设计** - 完美适配所有设备
- 🧮 **数学公式** - KaTeX支持
- 📊 **图表支持** - Mermaid图表
- 💻 **代码高亮** - 语法高亮显示
- 🚀 **性能优化** - 静态生成，加载快速

## 📂 项目结构

```
my-hugo/
├── archetypes/          # 内容模板
│   ├── default.md
│   └── posts.md
├── assets/             # 资源文件
│   ├── img/
│   └── js/
├── config/             # 配置文件
│   └── _default/
├── content/            # 内容文件
│   ├── posts/          # 博客文章
│   ├── _index.md       # 首页
│   ├── about.md        # 关于页面
│   ├── projects.md     # 项目页面
│   └── contact.md      # 联系页面
├── layouts/            # 自定义模板
├── public/             # 生成的静态文件
├── scripts/            # 辅助脚本
├── static/             # 静态资源
├── themes/             # 主题文件
└── package.json        # NPM配置
```

## 🔧 自定义配置

### 个人信息

在 `config/_default/params.toml` 中修改：

```toml
[author]
  name = "你的姓名"
  headline = "你的职业描述"
  bio = "个人简介"
  links = [
    { email = "mailto:your-email@example.com" },
    { github = "https://github.com/yourusername" },
  ]
```

### 网站信息

在 `config/_default/hugo.toml` 中修改：

```toml
title = "你的博客标题"
baseURL = "https://yourusername.github.io/your-repo"
```

## 🚀 部署

### GitHub Pages

1. 推送代码到GitHub仓库
2. 在仓库设置中启用GitHub Pages
3. 选择GitHub Actions作为部署源
4. 项目会自动构建和部署

### 其他平台

- **Netlify**: 连接GitHub仓库，自动部署
- **Vercel**: 导入GitHub项目，零配置部署
- **自托管**: 运行 `npm run build`，将 `public/` 目录部署到服务器

## 📊 SEO优化

项目已配置：

- ✅ 网站地图 (`sitemap.xml`)
- ✅ robots.txt 文件
- ✅ 结构化数据
- ✅ 社交媒体元标签
- ✅ Google Analytics集成

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。

## 📧 联系方式

- **邮箱**: pengyongqiang888@gmail.com
- **GitHub**: [@pengyongqiang666](https://github.com/pengyongqiang666)
- **博客**: [pengyongqiang666.github.io/my-hugo](https://pengyongqiang666.github.io/my-hugo)

---

**多思考，为防止小年痴呆！** 🧠✨