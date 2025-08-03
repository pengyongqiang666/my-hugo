# GitHub Pages 配置指南

本文档将指导您如何为 Hugo 博客配置 GitHub Pages 自动部署。

## 配置步骤

### 1. 更新 Hugo 配置

您需要修改 `config/_default/hugo.toml` 文件中的 `baseURL`：

```toml
# 将 YOUR_USERNAME 替换为您的 GitHub 用户名
# 将 YOUR_REPOSITORY_NAME 替换为您的仓库名称
baseURL = "https://YOUR_USERNAME.github.io/YOUR_REPOSITORY_NAME"
```

**示例：**
- 如果您的 GitHub 用户名是 `zhangsan`，仓库名是 `my-blog`
- 那么 baseURL 应该设置为：`https://zhangsan.github.io/my-blog`

### 2. 配置 GitHub 仓库设置

1. 打开您的 GitHub 仓库页面
2. 点击 **Settings**（设置）选项卡
3. 在左侧菜单中找到 **Pages** 选项
4. 在 **Source** 部分选择 **GitHub Actions**
5. 保存设置

### 3. 推送代码触发部署

```bash
# 提交并推送更改
git add .
git commit -m "配置 GitHub Pages 自动部署"
git push origin main
```

### 4. 检查部署状态

1. 在 GitHub 仓库页面，点击 **Actions** 选项卡
2. 您应该能看到 "Deploy Hugo site to Pages" 工作流正在运行
3. 等待工作流完成（通常需要 2-5 分钟）
4. 完成后，您的博客将在 `https://YOUR_USERNAME.github.io/YOUR_REPOSITORY_NAME` 可访问

## 工作流特点

### 自动化功能
- **自动触发**：每次推送到主分支时自动构建和部署
- **手动触发**：支持在 Actions 页面手动触发部署
- **增量构建**：只在内容更改时重新构建

### 技术特性
- 使用 Hugo Extended 版本（支持 SCSS）
- 启用 Dart Sass 支持
- 自动安装主题子模块
- 生产环境优化（压缩、清理）
- 自动配置相对 URL

## 故障排除

### 常见问题

1. **部署失败**
   - 检查 Actions 页面的错误日志
   - 确保所有子模块正确初始化
   - 验证 Hugo 配置文件语法

2. **页面无法访问**
   - 确认 baseURL 设置正确
   - 检查 GitHub Pages 设置中的源选择
   - 等待 DNS 传播（可能需要几分钟）

3. **样式丢失**
   - 确认 baseURL 末尾没有多余的斜杠
   - 检查主题资源是否正确构建

### 调试命令

```bash
# 本地测试构建
hugo server --environment production

# 检查子模块状态
git submodule status

# 更新子模块
git submodule update --init --recursive
```

## 后续维护

- **发布新文章**：创建新的 Markdown 文件并推送到主分支
- **更新主题**：定期更新 Blowfish 主题子模块
- **监控性能**：使用 Google Analytics 跟踪网站访问

## 参考链接

- [Hugo 官方文档](https://gohugo.io/documentation/)
- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [Blowfish 主题文档](https://blowfish.page/docs/)