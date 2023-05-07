# Solution to the preliminary competition questions of the algorithm group in the 2023 Huawei Embedded Software Competition
# 2023华为嵌入式软件大赛算法组初赛题目解决方案

## 1.Problem 问题
Please review the file "2023-华为嵌入式大赛软件赛-初赛赛题.pdf".

请查看文件"2023-华为嵌入式大赛软件赛-初赛赛题.pdf"。

## 2.My Solution 我的求解方法
There is no optimal solution in a competition, so this is just my poor solution idea, but it can solve for the result.

The basic model here is a weighted undirected graph, and it is already a fully connected graph. The method for solving the shortest path is the Dijkstra algorithm, even if there are multiple edges between two nodes in this problem. If the "number of channels" of each edge is converted into the "number of uses" of the edge, then adding conditions to determine the number of channels in Dijkstra can find the shortest possible feasible path.

For the strategy of adding edges, my method is relatively straightforward. If the shortest feasible path cannot be found, use the Dijkstra algorithm directly to find the shortest path without considering the number of channels, and then supplement the edges with 0 channels. If it is because the channels of the feasible business path cannot match (no common channel is idle), then check the channel status of all edges in the path and try to fill in as few edges as possible.

So the overall solution approach is based on the feasible path solved by the Dijkstra algorithm, and then the edge compensation and signal amplifier calculation are solved according to the situation. The complete process is as follows:

竞赛是没有最优解的，所以这只是我的不太好的求解思路，但是可以求解出结果。

这里的基本模型是一个有权无向图，而且已经是全联通的图。求解最短路径的方法就是Dijkstra算法，即使是本题存在两个节点间的多条边的情况。如果将每条边的“通道数”转化为边的“使用次数”，那么在Dijkstra中加入判断通道数的条件，就可以找到尽可能短的可行路线了。

对于加边的策略，我的方法比较直接。如果找不到尽可能短的可行路线时，就在不考虑通道数的情况下，直接用Dijkstra算法找最短路径，然后为通道数为0的边进行添补。如果是因为业务可行路径的通道匹配不上（没有一个共同的通道是空闲的），则检查路径中所有边的通道状况，尽可能少地补边。

所以整体的求解思路是基于Dijkstra算法求解出来的可行路径，然后根据状况进行补边和信号放大器的计算求解。完整的流程如下：

![微信图片_20230507132403](https://user-images.githubusercontent.com/59759272/236659376-5bee1278-4c56-4d95-9317-25b72de61b42.png)

## 3.Summary 总结
My method is relatively straightforward, so the final ranking is not very high, for everyone's reference only.

我的方法是比较直接的方法，所以最终排名也不是很高，仅供大家参考。
