v1: CFV对当前信息集policy求梯度，cfq是detach的
梯度是常数，本来loss不会传播梯度，但由于用熵正则化了，投影之后产生了耦合，所以仍然具有优化效果。

v2：CFV对历史信息集policy求梯度
和论文理论不是一回事

v3：直接将CFQ作为“当前信息集policy的梯度”，cfq不是detach的
loss通过cfq将梯度传播到历史信息集policy