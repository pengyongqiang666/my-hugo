# 虚拟窗口组件 (Virtual Window Component)

一个功能强大的Hugo partial模板组件，为博客页面提供可拖拽、可调整大小的虚拟窗口体验。

## 🗂️ 文件结构

```
layouts/partials/
├── virtual-window.html           # 主组件模板
├── virtual-window-assets.html    # CSS和JavaScript资源
└── README-virtual-window.md      # 本说明文档
```

## 🚀 快速开始

### 基础用法

```html
{{ partial "virtual-window.html" (dict "id" "my-window" "url" "/test.html") }}
```

### 完整参数

```html
{{ partial "virtual-window.html" (dict 
    "id" "demo-window"
    "title" "我的演示"
    "icon" "🎯"
    "url" "/demo.html"
    "width" "80%"
    "height" "70%"
    "minWidth" "400px"
    "minHeight" "300px"
    "buttonText" "🪟 打开窗口"
    "buttonStyle" "virtual-window-btn-primary"
) }}
```

## 📋 参数说明

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `id` | string | ✅ | 自动生成 | 窗口唯一标识符 |
| `title` | string | ❌ | "虚拟窗口" | 窗口标题 |
| `icon` | string | ❌ | "🪟" | 标题栏图标 |
| `url` | string | ✅ | 无 | 嵌入的页面URL |
| `width` | string | ❌ | "80%" | 初始宽度 |
| `height` | string | ❌ | "70%" | 初始高度 |
| `minWidth` | string | ❌ | "400px" | 最小宽度 |
| `minHeight` | string | ❌ | "300px" | 最小高度 |
| `buttonText` | string | ❌ | "🪟 打开虚拟窗口" | 按钮文字 |
| `buttonStyle` | string | ❌ | "virtual-window-btn-default" | 按钮样式类 |

## 🎨 按钮样式

- `virtual-window-btn-default` - 默认红色渐变
- `virtual-window-btn-primary` - 蓝紫色渐变  
- `virtual-window-btn-success` - 青绿色渐变

## ⌨️ 快捷键

- `ESC` - 关闭窗口
- `F11` - 切换全屏
- `Ctrl + M` - 最小化/恢复
- `双击标题栏` - 切换全屏

## 🔧 JavaScript API

```javascript
// 窗口控制
VirtualWindow.open('window-id');
VirtualWindow.close('window-id');
VirtualWindow.minimize('window-id');
VirtualWindow.toggleFullscreen('window-id');
VirtualWindow.focus('window-id');

// 状态查询
const state = VirtualWindow.getState('window-id');
const activeWindows = VirtualWindow.getActiveWindows();
```

## 💡 使用建议

1. **唯一ID**: 确保每个窗口ID在页面内唯一
2. **合理尺寸**: 根据内容设置合适的初始和最小尺寸
3. **URL安全**: 只嵌入可信任的内容
4. **移动优化**: 在移动设备上测试窗口体验
5. **性能考虑**: 避免单页面创建过多窗口

## 🔄 版本信息

- **版本**: 1.0.0
- **兼容性**: Hugo 0.88.0+
- **浏览器支持**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

## 📝 更新日志

### v1.0.0 (2025-01-13)
- ✅ 初始版本发布
- ✅ 完整的拖拽和调整大小功能
- ✅ 多种预设按钮样式
- ✅ 响应式设计支持
- ✅ 键盘快捷键支持
- ✅ JavaScript API

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

MIT License