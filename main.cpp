#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>
#include <unordered_map>
#include <stack>
#include <string>
#include <set>
#include <algorithm>

using namespace std;

/*************mode choose*************/ 
#define _DEBUG
#define _RELEASE

/*************macro definition*************/ 
#define PrintVar(var) cout << #var << " = " << var << endl;
#define LL long long 

/*************global variables*************/ 
// input variables
LL g_N;   // number of nodes
LL g_M;   // number of edges
LL g_T;   // number of business
LL g_P;   // number of edge's channels
LL g_D;   // maximum attenuation distance
struct BusinessInfo
{    
    LL src;
    LL dst;
    LL index;
    BusinessInfo(LL src_, LL dst_, LL index_) 
    {
       src = src_;
       dst = dst_;
       index = index_;
    }
};      // the strcut of bussiness information

vector<BusinessInfo> g_bussiness;  // Store the information about every bussiness

struct channel_state
{
    vector<LL> used_edges;
    LL total_remaining = 0;
    LL used_edges_num = 0;
};   // the strcut of the channel's state

struct BussinessResult
{
    LL channels_;
    vector<LL> path_edges_;
    vector<LL> larger_pos_;
};  // the strcut of the result of the Bussiness

// budget = g_addedge_num*1000000 + g_larger_num*100 + g_edge_num
LL g_addedge_num = 0;  // number of add edge
LL g_larger_num = 0;   // number of larger
LL g_edge_num = 0;     // number of edge

// output variables
vector<pair<LL,LL>> g_add_edge;                     // added edges
vector<vector<LL>> g_path(10000,vector<LL>(1,0));   // the path for every bussiness

/*************class definition*************/ 
class Edge 
{
public:
    LL src;        // the start of the edge
    LL dest;       // the end of the edge
    LL weight;     // the weight(length) of the edge
    LL remaining;  // the number of channels in the edge
    LL index;      // the index of the edge

    // record the uasge of the edge's channels
    vector<bool> channels{vector<bool>(80, false)}; 

    // construct the edge
    Edge(LL src, LL dest, LL weight, LL remaining, LL index) {
        this->src = src;
        this->dest = dest;
        this->weight = weight;
        this->remaining = remaining;
        this->index = index;
    }
}; // class Edge

class Graph 
{
public:
    LL V;               // The number of nodes
    vector<Edge>* adj;  // The neighors with its weight of every node

    void Create(LL V)
    {
        this->V = V;
        adj = new vector<Edge>[V];
    }

    // add edde
    void addEdge(LL u, LL v, LL w, LL r, LL ind) 
    {
        Edge e1(u, v, w, r, ind);
        adj[u].push_back(e1);
        Edge e2(v, u, w, r, ind);
        adj[v].push_back(e2);
    }

}; // class Graph 

/*************global varibles 2*************/ 
Graph g_graph;

/**************functionnal func**************/ 
// Calculate the total budget
LL CalculateTotalBudget()
{
    return g_addedge_num * 1000000 + g_larger_num * 100 + g_edge_num;
}

// Calculate the buget for one bussiness solution
LL CalculateBudget(LL addedge_num_, LL larger_num_, LL edge_num_)
{
    return addedge_num_ * 1000000 + larger_num_ * 100 + edge_num_;
}

// Print the state of Graph
void PrintGraph(Graph graph_)
{
    cout << "PrintGraph:\n--------------\n" ;
    cout << "u v w r ind\n";
    for(LL node_index = 0; node_index < g_N; node_index++)
    {
        cout << "node = " << node_index << endl;
        for (auto e : graph_.adj[node_index]) 
        {
            cout << e.src << " " << e.dest << " " << e.weight \
            << " " << e.remaining << " " << e.index << endl;
        }
    }
    
    cout << "--------------\n";
}

// Get input parameters about the graph
void GetGlobalInput()
{
    cin >> g_N >> g_M >> g_T >> g_P >> g_D;
}

// Get specific information about graph to create the graph
void CreateGraph()
{
    g_graph.Create(g_N);
    // edge
    for (LL edge = 0; edge < g_M; edge++)
    {
        LL s, t, d;
        cin >> s >> t >> d;
        g_graph.addEdge(s, t, d, g_P, edge);
    }
    // business
    for(LL bussiness = 0; bussiness < g_T; bussiness++)
    {
        LL s,t;
        cin >> s >> t;
        BusinessInfo tmp(s,t,bussiness);
        g_bussiness.push_back(tmp);
    }
}

// Using DFS algorithm to find all the paths
void DFS(Graph graph_, LL cur_, LL dest_, LL cur_length_, \
        vector<vector<LL>>& cur_path_, vector<vector<vector<LL>>>& all_path_, \
        set<LL>& visited_)
{
    // find one way to dest
    if (cur_ == dest_)
    {   
        all_path_.push_back(cur_path_);
        return;
    }

    // traverse all the neighbors for the current node
    for (auto e : graph_.adj[cur_]) 
    {
        if(visited_.find(e.index)==visited_.end()) 
        {
            // format:{prenode, node, length, total_length, index of the edge}
            vector<LL> current_state;    
            current_state.push_back(cur_);                  // prenode
            current_state.push_back(e.dest);                // node
            current_state.push_back(e.weight);              // length
            current_state.push_back(cur_length_+e.weight);  // length
            current_state.push_back(e.index);               // index of the edge

            visited_.insert(e.index);

            cur_path_.push_back(current_state);

            DFS(graph_, e.dest, dest_, cur_length_+e.weight, cur_path_, \
                all_path_, visited_);

            visited_.erase(e.index);
            cur_path_.pop_back();
        }
    }
}

// produce the result of the bussiness in formal format
void DijsktraResult(BussinessResult& br_, LL index_)
{
    vector<LL> bussiness_result;
    bussiness_result.push_back(br_.channels_);          // channel index (not decide)
    bussiness_result.push_back(br_.path_edges_.size()); // number of edges
    bussiness_result.push_back(br_.larger_pos_.size()); // number of larger
    for(LL path_edge = 0; path_edge < br_.path_edges_.size(); path_edge++)
    {
        bussiness_result.push_back(br_.path_edges_[path_edge]);
    }
    for(LL larger_pos = 0; larger_pos < br_.larger_pos_.size(); larger_pos++)
    {
        bussiness_result.push_back(br_.larger_pos_[larger_pos]);
    }
    g_path[index_] = bussiness_result;

#ifdef _DEBUG
    cout << "The result of the current bussiness:\n";
    for(LL i = 0; i < bussiness_result.size(); i++)
    {
        cout << bussiness_result[i] << " ";
    }
    cout << endl;
#endif
}

// Compare two paths to choose when no available channel for current path
int DijkstraCompare(Graph& graph_, LL src_, LL dest_,\
                    unordered_map<LL, vector<LL>> parent_,\
                    vector<channel_state>& channels_used_) 
{
    // Calculate the channel state
    LL node = dest_;
    LL current_path_length = 0; 
    while (node != src_)
    {
        LL par = parent_[node][0];
        // traverse all the neighbors of the node
        for (auto e : graph_.adj[node]) 
        {
            // if the edge is in the availble path
            if (e.index == parent_[node][2])
            {
                current_path_length += e.weight;
                // check every channel of the edge
                for(LL channel_index = 0; channel_index < g_P; ++channel_index)
                {
                    // if used
                    if(e.channels[channel_index])
                    {
                        channels_used_[channel_index].total_remaining += e.remaining;
                        channels_used_[channel_index].used_edges.push_back(e.index);
                        channels_used_[channel_index].used_edges_num += 1;
                    }
                }
                break;
            }
        }
        node = par;
    }

    // find the minimum used_edges's channels 
    auto cmp = [](const channel_state a, channel_state b)
    {
        if (a.used_edges_num < b.used_edges_num)
        {
            return true;
        }
        else if (a.used_edges_num > b.used_edges_num)
        {
            return false;
        }
        else if (a.used_edges_num == b.used_edges_num)
        {
            return a.total_remaining < b.total_remaining;
        }
        return false;
    };

    sort(channels_used_.begin(), channels_used_.end(), cmp);

    // calculate the number of needed to added edges for the current paths (mostly are not the shortest paths)
    int current_path_needed_edges = channels_used_[0].used_edges.size();

    /* find the shortest path to compare */
    priority_queue<pair<LL, LL>, vector<pair<LL, LL>>, greater<pair<LL, LL>>> pq; 
    vector<LL> dist(graph_.V, INT_MAX);
    unordered_map<LL, vector<LL>> parent;
    dist[src_] = 0;

    pq.push({0, src_});

    // Find the shortest path to the every node
    while (!pq.empty()) 
    {
        LL u = pq.top().second;
        pq.pop();

        // Traverse all neighbors
        for (auto e : graph_.adj[u]) 
        {
            LL v = e.dest;
            LL weight = e.weight;

            LL newDist = dist[u] + weight;

            // if channels is not empty and new path is shorter
            if (newDist < dist[v]) 
            {
                dist[v] = newDist;

                // upgrade the parent of the shorter path
                parent[v] = {u, weight, e.index};      

                pq.push({newDist, v});  
            }  
        }
    }

    // If the current path is the shortest path, end comparing
    if (parent == parent_) return 1;

    LL shortset_path_length = 0;

    // find the shortest path
    int shortest_path_needed_edges = 0;
    node = dest_;
    while (node != src_)
    {
        LL par = parent[node][0];

        // traverse all the neighbors of the node
        for (auto& e : graph_.adj[node]) 
        {
            // if the edge is in the shortest path
            if (e.index == parent[node][2])
            {
                shortset_path_length += e.weight;
                // need to add edge
                if(e.remaining == 0)
                {
                    shortest_path_needed_edges ++;
                }
                break;
            }
        }
        node = par;
    }

#ifdef _DEBUG
    PrintVar(shortest_path_needed_edges)
    PrintVar(current_path_needed_edges)
#endif

    // calculate the budgets of two paths roughly
    LL current_path_budget  = CalculateBudget(current_path_needed_edges,\
                                current_path_length/g_D , parent_.size());
    LL shortest_path_budget = CalculateBudget(shortest_path_needed_edges,\
                                shortset_path_length/g_D , parent.size());
    
#ifdef _DEBUG
    PrintVar(current_path_budget)
    PrintVar(shortest_path_budget)
#endif

    // compare the edges
    int res;
    if (shortest_path_budget >= current_path_budget)
    {
        res =  1; // chose the current path
    }
    else
    {
        res =  0; // chose the shortest path
    }
    return res;
}

// Find the shortest path for the bussiness with Dijkstra algorithm
// state_flag_ : 0---normal ; 1---no path
void Dijkstra(Graph& graph_, LL src_, LL dest_, int state_flag_, LL index_) 
{
    // small top heap
    // pair: {distance : node}
    priority_queue<pair<LL, LL>, vector<pair<LL, LL>>, greater<pair<LL, LL>>> pq; 

    // storage the shortest length of every node from the start node
    vector<LL> dist(graph_.V, INT_MAX); 
    
    // storage the pre node, the length to the pre node and the index of the edge in the shortest path
    // node : {pre node, length, index of edge}
    unordered_map<LL, vector<LL>> parent;

    // the distance to the itself of node is 0
    dist[src_] = 0;

    pq.push({0, src_});

    // Find the shortest path to the every node
    while (!pq.empty()) 
    {
        LL u = pq.top().second;
        pq.pop();

        // Traverse all neighbors
        for (auto e : graph_.adj[u]) 
        {
            LL v = e.dest;
            LL weight = e.weight;

            LL newDist = dist[u] + weight;

            if (state_flag_==0)
            {
                // if channels is not empty and new path is shorter
                if (e.remaining>0 && newDist<dist[v]) 
                {
                    dist[v] = newDist;

                    // upgrade the parent of the shorter path
                    parent[v] = {u, weight, e.index};      

                    pq.push({newDist, v});  
                }
            }
            else // not considering the remaining of edges
            {
                // if channels is not empty and new path is shorter
                if (newDist < dist[v]) 
                {
                    dist[v] = newDist;

                    // upgrade the parent of the shorter path
                    parent[v] = {u, weight, e.index};      

                    pq.push({newDist, v});  
                }
            }
                
        }
    }

    // Cannot get to destination
    if (dist[dest_] == INT_MAX) 
    {
    #ifdef _DEBUG
        cout << "No path found from " << src_ << " to " << dest_ << "." << endl;
    #endif

        // find the shortest path without considering the remaining of the edge
        Dijkstra(graph_, src_, dest_, 1, index_);

        return;
    }

    // no path -> find the shortest path to add edges which the reamining = 0
    // no available channel -> find the shortest path to compare with the other path
    if(state_flag_ == 1)
    {
        LL node = dest_;
        while (node != src_)
        {
            LL par = parent[node][0];

            // traverse all the neighbors of the node
            for (auto& e : graph_.adj[node]) 
            {
                // if the edge is in the shortest path
                if (e.index == parent[node][2])
                {
                    // need to add edge
                    if(e.remaining == 0)
                    {
                        // count
                        g_addedge_num += 1;

                        // record added edge
                        g_add_edge.push_back(make_pair(par, node));

                    #ifdef _DEBUG
                        cout << "need to add edge becasue of remaining = 0." << endl;
                        cout << "Added edge = " << g_M+g_addedge_num-1 << endl;
                    #endif

                        // add edge in graph -> just change the edge's index and channel state
                        for (auto& ee : graph_.adj[node]) 
                        {
                            if (ee.index == parent[node][2])  
                            {                                
                                ee.index     = g_M + g_addedge_num - 1;
                                ee.remaining = g_P;
                                ee.channels  = {vector<bool>(g_P, false)};
                                break;
                            }
                        }
                        for (auto& ee : graph_.adj[par]) 
                        {
                            if (ee.index == parent[node][2]) 
                            {
                                ee.index     = g_M + g_addedge_num - 1;
                                ee.remaining = g_P;
                                ee.channels  = {vector<bool>(g_P, false)};
                                break;
                            }
                        }

                        // update path
                        parent[node][2] = g_M + g_addedge_num - 1;
                    }
                    break;
                }
            }
            node = par;
        }
    }

    // Calculate the channel state
    LL node = dest_;
    vector<bool> if_used_channel(g_P,false);
    while (node != src_)
    {
        LL par = parent[node][0];

        // traverse all the neighbors of the node
        for (auto e : graph_.adj[node]) 
        {
            // if the edge is in the available path
            if (e.index == parent[node][2])
            {
                for(LL channel_index = 0; channel_index < g_P; ++channel_index)
                {
                    if_used_channel[channel_index] = if_used_channel[channel_index] | e.channels[channel_index];
                }
                break;
            }
        }
        node = par;
    }

    // Find the available channel for every edge
    LL channels = -1;
    for (LL channel_index = 0; channel_index < g_P; ++channel_index) 
    {
        // choose the smallest channel
        if (if_used_channel[channel_index] == false)
        {
            channels = channel_index;
            break;
        }
    }

#ifdef _DEBUG
    cout << "the channel result:\n";
    for(auto i : if_used_channel)
    {
        cout << i << " ";
    }
    cout << endl;
#endif

    // No channels available => add edges
    if( channels == -1 )
    {
    #ifdef _DEBUG
        cout << "No channels available." << endl;
    #endif

        channel_state tmp;
        vector<channel_state> channels_used(g_P, tmp);

        // compare the current path and the shortest path
        if ( DijkstraCompare(graph_, src_, dest_, parent, channels_used)==1 ) 
        {
            // chose the current path
        #ifdef _DEBUG
            cout << "chose the current path" << endl;
        #endif

            // add edges & redefine the channel
            node = dest_;
            vector<bool> if_used_channel_2(g_P,false);
            while (node != src_)
            {
                LL par = parent[node][0];
                
                // traverse all the neighbors of the node
                for (auto& e : graph_.adj[node]) 
                {
                    // if the edge is in the path
                    if (e.index == parent[node][2])
                    {
                        for(auto edge_needed : channels_used[0].used_edges)
                        {
                            // need to add edge
                            if (edge_needed == e.index)
                            {
                                // count
                                g_addedge_num += 1;

                                // record added edges
                                g_add_edge.push_back(make_pair(par, node));

                            #ifdef _DEBUG
                                cout << "need to add edge" << endl;
                                cout << "Added edge = " << g_M+g_addedge_num-1 << endl;
                            #endif

                                // add edge in graph -> just change the edge's index and channel state
                                for (auto& ee : graph_.adj[node]) 
                                {
                                    if (ee.index == parent[node][2])  
                                    {
                                        ee.index     = g_M + g_addedge_num - 1;
                                        ee.remaining = g_P;
                                        ee.channels  = {vector<bool>(g_P, false)};
                                        break;
                                    }
                                }
                                for (auto& ee : graph_.adj[par]) 
                                {
                                    if (ee.index == parent[node][2]) 
                                    {
                                        ee.index     = g_M + g_addedge_num - 1;
                                        ee.remaining = g_P;
                                        ee.channels  = {vector<bool>(g_P, false)};
                                        break;
                                    }
                                }

                                // update path
                                parent[node][2] = g_M + g_addedge_num - 1;

                                break;
                            }
                        }

                        for(LL i = 0; i < g_P; ++i)
                        {
                            if_used_channel_2[i] = if_used_channel_2[i] | e.channels[i];
                        }

                        break;
                    }
                }
                node = par;
            }
            for (LL i = 0; i < g_P; ++i) 
            {
                // choose the smallest channel
                if (if_used_channel_2[i] == false)
                {
                    channels = i;
                    break;
                }
            }
        }
        else    // chose the shortest path
        {
        #ifdef _DEBUG
            cout << "choose the shortest path" << endl;
        #endif

            Dijkstra(graph_, src_, dest_, 1, index_);
            return;
        }
    }

    // backtracking to find the available path
    // & update the information of the edges(remaining & channels[])
    stack<LL> path;                  // record the shortest path
    stack<vector<LL>> path_length;   // record the {pre node, length, index of edge}
    node = dest_;
    path.push(node);                        // add the destination node mannully
    
    while (node != src_) 
    {
        LL par = parent[node][0];
        path_length.push(parent[node]);

        // deal with the neighbors of the current node
        for (auto& e : graph_.adj[node]) 
        {
            if (e.index == parent[node][2])
            {
                e.remaining -= 1;
                e.channels[channels] = true;
                break;
            }
        }
        for (auto& e : graph_.adj[par]) 
        {
            if (e.index == parent[node][2])
            {
                e.remaining -= 1;
                e.channels[channels] = true;
                break;
            }
        }
        
        node = par; // move to the parent node
        path.push(node);
    }

    // evaluate if add larger 
    // & calculte the number of edges
    // & find the path in edges'index format
    LL signal_intensity = g_D; // the intensity of the signal of the bussiness
    vector<LL> path_edges;     // available path in edge format
    vector<LL> larger_pos;     // store the larger position
    while(!path_length.empty())
    {
        vector<LL> cur = path_length.top();
        signal_intensity -= cur[1];

        path_edges.push_back(cur[2]);
        // if signal_intensity is not enough
        if (signal_intensity <= 0)
        {
            larger_pos.push_back(cur[0]);
            signal_intensity = g_D - cur[1];

        #ifdef _DEBUG
            g_larger_num += 1;
        #endif
        }

        path_length.pop();
    }

#ifdef _DEBUG
    g_edge_num += path_edges.size();
#endif

    // produce the result of the bussiness in formal format
    BussinessResult tmp;
    tmp.channels_   = channels;
    tmp.path_edges_ = path_edges;
    tmp.larger_pos_ = larger_pos;
    DijsktraResult(tmp, index_);

#ifdef _DEBUG
    // Print the Shortest Path
    cout << "Shortest path (Point Format) from " << src_ << " to " << dest_ << ": ";
    string shortest_path_string = "";
    while (!path.empty()) 
    {
        LL cur = path.top();
        shortest_path_string += to_string(cur) + " ";
        path.pop();
    }
    cout << shortest_path_string  << endl;

    cout << "Shortest path (Edge Format) from " << src_ << " to " << dest_ << ": ";
    for(auto i : path_edges)
    {
        cout << i << " ";
    }
    cout << endl;
#endif
}

// Find shortest path by using Dijkstra
void DijkstraFindShortestPath(LL src_, LL dest_, LL index_)
{
    Dijkstra(g_graph, src_, dest_, 0, index_);
}

// Find all the paths by using DFS
void DFSFindAllPaths(LL src_, LL dest_)
{
    vector<vector<LL>> cur_path;
    vector< vector<vector<LL>> > all_paths;
    set<LL> visited;

    DFS(g_graph, src_, dest_, 0, cur_path, all_paths, visited);

#ifdef _DEBUG
    // every path
    for(auto j : all_paths)
    {
        cout << "path:" << endl;
        // every edge
        for(auto k: j)
        {
            for(auto l : k)
            {
                cout << l << " ";
            }
            cout << endl;
        }
    }
#endif
}

// Deal All Business
void DealBusiness()
{
    LL kk = 1;

    auto cmp = [](const BusinessInfo a, BusinessInfo b)
    {
        if (a.src < b.src) return true;
        else if (a.src > b.src) return false;
        else
        {
            if (a.dst < b.dst) return true;
            else if (a.dst > b.dst) return false;
            else return false;
        }
    };

    sort(g_bussiness.begin(), g_bussiness.end(),cmp);

    for (auto i : g_bussiness)
    {
    #ifdef _DEBUG
        cout << "(" << kk++ << ") ";
    #endif
    
        // Dijstra
        DijkstraFindShortestPath(i.src, i.dst, i.index);
    }
}

// output with the formal law
void FormalOutput()
{
#ifdef _DEBUG
    cout << "Formal Output------------------" << endl;
#endif
    // added edges
    cout << g_addedge_num << endl;
    for(LL i = 0; i < g_add_edge.size(); i++)
    {
        cout << g_add_edge[i].first << " " << g_add_edge[i].second << endl;
    }

    // the result of every bussiness
    for(LL i = 0; i < g_T; ++i)
    {
        for(auto j : g_path[i])
        {
            cout << j << " ";       
        }
        cout << endl;
    }

#ifdef _DEBUG
    cout << "Formal Output------------------" << endl;
#endif
}

void TestOutput()
{
    cout << "Test Output------------------" << endl;
    cout << "g_larger_num = " << g_larger_num << endl;
    cout << "g_edge_num = "   << g_edge_num << endl;
    cout << "Total budget = " << CalculateTotalBudget() << endl;
    cout << "Test Output------------------" << endl;
}

/**************main function**************/ 
int main()
{
#ifdef _DEBUG
    cout << "Beginning of main()." << endl;
#endif

    GetGlobalInput();

    CreateGraph();

    DealBusiness();

#ifdef _DEBUG
    TestOutput();
#endif
    
#ifdef _RELEASE
    FormalOutput();
#endif

#ifdef _DEBUG
    cout << "End of main()." << endl;
#endif

    return 0;
}