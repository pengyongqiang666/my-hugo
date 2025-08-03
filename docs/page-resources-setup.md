# Hugo 页面资源配置指南

本文档说明如何配置 Hugo 使 HTML 文件可以作为页面资源被访问。

## 问题背景

默认情况下，Hugo 将 HTML 文件视为页面内容（resource type: page），不会将其发布为可直接访问的静态资源。当使用虚拟窗口短代码尝试访问同目录下的 HTML 文件时，会遇到 404 错误。

## 解决方案

### 1. 配置内容类型 (Content Types)

在 `config/_default/hugo.toml` 中添加以下配置：

```toml
# 内容类型配置 - 允许 HTML 页面资源被发布
[contentTypes]
  "text/asciidoc" = {}
  "text/markdown" = {}
  "text/org" = {}
  "text/pandoc" = {}
  "text/rst" = {}
  # 移除 text/html 使其能够作为普通资源被发布
```

**关键点**：通过不包含 `"text/html" = {}` 配置，Hugo 会将 HTML 文件的资源类型从 `page` 改为 `text`，使其能够被发布为可访问的资源。

### 2. 文件组织结构

将 HTML 文件放在与文章相同的目录下：

```
content/
└── posts/
    └── your-article/
        ├── index.md        # 文章内容
        ├── test.html       # HTML 资源文件
        ├── featured.svg    # 特色图片
        └── background.jpg  # 背景图片
```

### 3. 短代码配置

在虚拟窗口短代码中使用相对路径：

```markdown
{{< virtual-window 
    id="demo-window" 
    title="演示窗口"
    icon="🪟"
    url="test.html"  <!-- 使用相对路径，不是 /test.html -->
    buttonText="🪟 打开演示" 
    width="75%"
    height="65%"
>}}
```

## 访问路径

配置完成后，HTML 文件的访问路径为：
- 文章路径：`/posts/your-article/`
- HTML 资源：`/posts/your-article/test.html`

## 原理说明

### 默认行为
Hugo 默认的 `contentTypes` 配置包含了所有内容格式：
```toml
[contentTypes]
  "text/asciidoc" = {}
  "text/html" = {}      # 这一行使 HTML 被视为页面内容
  "text/markdown" = {}
  "text/org" = {}
  "text/pandoc" = {}
  "text/rst" = {}
```

### 修改后的行为
移除 `"text/html" = {}` 后：
- HTML 文件的资源类型变为 `text`
- Hugo 会将其作为普通静态资源发布
- 可以通过页面资源路径直接访问

## 优势

1. **保持组织性**：资源文件与文章保持在同一目录
2. **版本控制友好**：相关文件集中管理
3. **部署简单**：无需额外配置静态资源路径
4. **维护方便**：文章和资源的生命周期一致

## 注意事项

- 重启 Hugo 服务器以应用配置更改
- 确保 HTML 文件名唯一，避免路径冲突
- 在生产环境中测试配置的有效性

## 参考文档

- [Hugo Page Resources](https://gohugo.io/content-management/page-resources/)
- [Hugo Content Types](https://gohugo.io/configuration/content-types/)
- [Hugo Page Bundles](https://gohugo.io/content-management/page-bundles/)