from src.vector_db import semantic_search

def main():
    """
    主交互界面
    """
    print("=" * 60)
    print("        兴化方言语义检索系统（最终增强版）")
    print("=" * 60)
    print("支持功能：")
    print("1. 普通话查方言（如“爸爸”→“郎罢”）")
    print("2. 自然语言提问（如“我膝盖疼用方言怎么说”）")
    print("3. 同义词查询（如“用餐”→“吃饭”）")
    print("输入 q 或 quit 退出系统\n")

    while True:
        # 获取用户输入
        user_input = input("请输入查询：").strip()
        
        # 退出逻辑
        if user_input.lower() in ["q", "quit", "exit"]:
            print("再见！")
            break
        
        # 空输入处理
        if not user_input:
            print("查询内容不能为空，请重新输入！")
            continue
        
        # 执行检索
        try:
            results = semantic_search(user_input, top_k=5)
        except Exception as e:
            print(f"检索出错：{e}")
            continue
        
        # 打印结果
        if not results:
            print("未找到匹配的方言词条，请更换查询词重试！")
            continue
        
        print("\n--- 检索结果（按相似度排序） ---")
        for i, (entry_id, similarity, detail) in enumerate(results, 1):
            print(f"\n{i}. 【相似度：{similarity:.3f}】")
            print(f"   方言词：{detail['方言词']}")
            print(f"   简易发音：{detail['简易发音']}")
            print(f"   标准发音：{detail['标准发音']}")
            print(f"   释义注释：{detail['释义注释']}")
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()