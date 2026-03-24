# demo.py（方案二版本）
from src.vector_db_v2 import semantic_search_v2  # 改成导入v2的函数

def interactive_search():
    print("="*60)
    print("          兴化方言语义检索系统（方案二：全文本拼接）")
    print("="*60)
    print("使用说明：")
    print("1. 输入查询文本（如“爸爸”“吃饭 方言”），按回车检索；")
    print("2. 输入“q”或“quit”退出程序；")
    print("3. 检索结果按相似度排序，得分越高越匹配（0-1）；")
    print("="*60 + "\n")

    while True:
        query = input("请输入查询内容（输入q退出）：").strip()
        if query.lower() in ["q", "quit", "exit"]:
            print("退出检索系统，再见！")
            break
        if not query:
            print("查询内容不能为空，请重新输入！")
            continue
        
        try:
            results = semantic_search_v2(query, top_k=10)  # 改成调用v2的函数
        except Exception as e:
            print(f"检索出错：{e}")
            continue
        
        if not results:
            print("未找到匹配的方言词条，请更换查询词重试！")
            continue
        
        print(f"\n共找到 {len(results)} 条匹配结果（按相似度排序）：")
        print("-"*80)
        for idx, (entry_id, similarity, detail) in enumerate(results, 1):
            print(f"\n【结果 {idx}】")
            print(f"  词条ID：{entry_id}")
            print(f"  相似度：{similarity:.4f}")
            print(f"  方言词：{detail['方言词']}")
            print(f"  简易发音：{detail['简易发音']}")
            print(f"  标准发音：{detail['标准发音']}")
            print(f"  释义注释：{detail['释义注释']}")
            print("-"*80)

if __name__ == "__main__":
    interactive_search()