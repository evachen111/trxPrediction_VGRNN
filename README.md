# trxPrediction-VGRNN

## Transaction Prediction in the Bitcoin Network

### Dataset: 

The graph data encompasses information spanning from 03/01/2009 to 14/12/2020. As time progressed, the graph's node and edge counts expanded from 0 to a peak of 629,204 nodes and 1,014,634 edges, respectively. The edges between nodes symbolize transactions between entities within the network. The edges encompass key attributes, including "n\_tx" representing the number of transactions, and "value" indicating the transaction's value. In general, the blockchain network is vast and dynamic. Every day, over 90\% of the nodes and edges in the network are added or removed, causing it to constantly evolve and produce sparse graphs of daily snapshots. Address clustering needs to be done under certain heuristics.

### Task Definition:

Considering the dynamic graph  $G = \lbrace G^{(1)}, G^{(2)},..., G^{(T)} \rbrace$, $G^{(t)} = (V^{(t)}, E^{(t)})$ is the graph snapshot at time $t$. $V^{(t)}$ and $E^{(t)}$ are the corresponding node and edge sets and the cardinality of both $V^{(t)}$ and $E^{(t)}$, i.e. the number of nodes $N_{t}$ and the number of edges $E_{t}$ can change over time. Being an essential representation of graphs, the variable-length adjacency matrix sequence $A = \lbrace A^{(1)}, A^{(2)},..., A^{(T)} \rbrace$, where $A^{(t)}$ is a $N_{t}\times N_{t}$ matrix. When considering node attributes, the variable-length node attribute sequence $X = \lbrace X^{(1)}, X^{(2)},..., X^{(T)} \rbrace$, where $X^{(t)}$ is a $N_{t}\times M$ matrix. In the dynamic graph embedding literature, the term link prediction has different definitions. In this work, it is this way here: given observed snapshots of a dynamic graph $G^{(T)}$, the task is to predict edges in $G^{(T+1)}$ with the time interval to be one day, one week, one month, or one year.


### Model: VGRNN
```
@inproceedings{hajiramezanali2019variational,
  title={Variational graph recurrent neural networks},
  author={Hajiramezanali, Ehsan and Hasanzadeh, Arman and Narayanan, Krishna and Duffield, Nick and Zhou, Mingyuan and Qian, Xiaoning},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10700--10710},
  year={2019}
}
```

## Code

snippet for predictions at one-day intervals

### Requirements
CUDA==9.0.176
Python==2.7.12
networkx==2.2
scipy==1.1.0
torch==1.0.0      
torch-cluster==1.2.3      
torch-geometric==1.0.2      
torch-scatter==1.1.1      
torch-sparse==0.2.3      
torch-spline-conv==1.0.5      
torchvision==0.2.1

