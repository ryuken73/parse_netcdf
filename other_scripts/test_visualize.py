# import plotly.express as px
# import pandas as pd

# # 데이터 예제 (단어, 빈도수)
# data = pd.DataFrame({
#     "단어": ["AI", "머신러닝", "딥러닝", "데이터 분석", "자연어처리", "추천 시스템", "강화학습"],
#     "빈도": [50, 40, 35, 30, 25, 20, 15]
# })

# # 트리맵 생성
# fig = px.treemap(data, path=["단어"], values="빈도", title="텍스트 빈도 Treemap")
# fig.show()

# import networkx as nx
# import matplotlib.pyplot as plt

# G = nx.Graph()
# edges = [("AI", "머신러닝"), ("머신러닝", "딥러닝"), ("딥러닝", "데이터"), ("데이터", "분석")]

# G.add_edges_from(edges)
# nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
# plt.show()

# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt

# data = np.array([[3, 2, 1], [5, 4, 3], [7, 6, 5]])
# sns.heatmap(data, annot=True, cmap="coolwarm")

# plt.show()

# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# data = pd.DataFrame({"리뷰": ["좋아요", "별로예요", "최고예요", "싫어요"], "점수": [0.8, -0.5, 0.9, -0.7]})
# sns.barplot(x="리뷰", y="점수", data=data, palette="coolwarm")
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({"날짜": ["2024-01", "2024-02", "2024-03"], "AI 관련 뉴스 개수": [10, 25, 40]})
data["날짜"] = pd.to_datetime(data["날짜"])

plt.plot(data["날짜"], data["AI 관련 뉴스 개수"], marker="o")
plt.title("시간별 AI 관련 뉴스 개수 변화")
plt.show()
