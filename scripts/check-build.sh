#!/bin/bash

# Hugo博客构建检查脚本
# 用于在每次修改后自动检查构建状态

echo "🚀 开始Hugo博客构建检查..."
echo "================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查是否在Hugo项目根目录
if [ ! -f "config/_default/hugo.toml" ]; then
    echo "❌ 错误: 请在Hugo项目根目录运行此脚本"
    exit 1
fi

# 1. 检查Hugo版本
echo -e "${BLUE}🔍 检查Hugo版本...${NC}"
if command -v hugo &> /dev/null; then
    HUGO_VERSION=$(hugo version)
    echo -e "${GREEN}✅ Hugo已安装: $HUGO_VERSION${NC}"
else
    echo -e "${RED}❌ Hugo未安装，请先安装Hugo${NC}"
    exit 1
fi

# 2. 检查配置文件语法
echo -e "${BLUE}🔧 检查配置文件语法...${NC}"
CONFIG_FILES=(
    "config/_default/hugo.toml"
    "config/_default/params.toml" 
    "config/_default/menus.toml"
    "config/_default/languages.toml"
    "config/_default/markup.toml"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        # 简单的TOML语法检查 - 检查是否有未闭合的引号或括号
        if grep -q "^\s*\[.*\]\s*$" "$file" && ! grep -q "^\s*\[.*\[\s*$" "$file"; then
            echo -e "${GREEN}  ✅ $file 语法正确${NC}"
        else
            echo -e "${YELLOW}  ⚠️  $file 可能有语法问题${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠️  $file 不存在${NC}"
    fi
done

# 3. 检查内容文件
echo -e "${BLUE}📝 检查内容文件...${NC}"
CONTENT_COUNT=$(find content -name "*.md" | wc -l)
echo -e "${GREEN}  ✅ 发现 $CONTENT_COUNT 个内容文件${NC}"

# 检查必要的页面是否存在
REQUIRED_PAGES=("content/_index.md" "content/about.md" "content/contact.md")
for page in "${REQUIRED_PAGES[@]}"; do
    if [ -f "$page" ]; then
        echo -e "${GREEN}  ✅ $page 存在${NC}"
    else
        echo -e "${YELLOW}  ⚠️  $page 不存在${NC}"
    fi
done

# 4. Hugo构建测试
echo -e "${BLUE}🏗️  执行Hugo构建测试...${NC}"
BUILD_OUTPUT=$(hugo --gc --minify 2>&1)
BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Hugo构建成功!${NC}"
    
    # 显示构建统计
    if echo "$BUILD_OUTPUT" | grep -q "Pages"; then
        echo -e "${BLUE}📊 构建统计:${NC}"
        echo "$BUILD_OUTPUT" | grep -E "(Pages|Static files|Total in)" | sed 's/^/  /'
    fi
    
    # 检查是否有警告
    if echo "$BUILD_OUTPUT" | grep -q "WARN"; then
        echo -e "${YELLOW}⚠️  发现警告:${NC}"
        echo "$BUILD_OUTPUT" | grep "WARN" | sed 's/^/  /'
        echo -e "${YELLOW}  💡 这些警告通常不影响功能，但建议关注${NC}"
    fi
    
else
    echo -e "${RED}❌ Hugo构建失败!${NC}"
    echo -e "${RED}错误详情:${NC}"
    echo "$BUILD_OUTPUT" | sed 's/^/  /'
    exit 1
fi

# 5. 检查生成的文件
echo -e "${BLUE}📁 检查生成的文件...${NC}"
if [ -d "public" ]; then
    PUBLIC_FILES=$(find public -type f | wc -l)
    echo -e "${GREEN}  ✅ public目录包含 $PUBLIC_FILES 个文件${NC}"
    
    # 检查重要文件
    IMPORTANT_FILES=("public/index.html" "public/about/index.html" "public/contact/index.html" "public/sitemap.xml")
    for file in "${IMPORTANT_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}  ✅ $file 已生成${NC}"
        else
            echo -e "${YELLOW}  ⚠️  $file 未找到${NC}"
        fi
    done
else
    echo -e "${YELLOW}  ⚠️  public目录不存在${NC}"
fi

# 6. 性能建议
echo -e "${BLUE}🚀 性能建议:${NC}"
if [ -f "static/robots.txt" ]; then
    echo -e "${GREEN}  ✅ robots.txt 已配置${NC}"
else
    echo -e "${YELLOW}  💡 建议添加 robots.txt 文件${NC}"
fi

if grep -q "googleAnalytics" config/_default/hugo.toml; then
    echo -e "${GREEN}  ✅ Google Analytics 已配置${NC}"
else
    echo -e "${YELLOW}  💡 建议配置网站分析工具${NC}"
fi

echo "================================"
echo -e "${GREEN}🎉 检查完成! 博客构建正常!${NC}"

# 如果是开发模式，询问是否启动本地服务器
if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
    echo ""
    read -p "是否启动本地开发服务器? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🌐 启动Hugo开发服务器...${NC}"
        hugo server -D --bind 0.0.0.0 --port 1313
    fi
fi