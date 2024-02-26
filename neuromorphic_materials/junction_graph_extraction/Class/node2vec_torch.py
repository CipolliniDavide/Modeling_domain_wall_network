import torch
from torch_geometric.nn import Node2Vec


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model()
    acc = model.test(
        z[data.train_mask],
        data.y[data.train_mask],
        z[data.test_mask],
        data.y[data.test_mask],
        max_iter=150,
    )
    return acc


def node2vec_call(
    data,
    device,
    node2vec_par={
        "embedding_dim": 2,
        "walk_length": 20,
        "                    context_size": 10,
        "walks_per_node": 10,
        "num_negative_samples": 1,
        "p": 1,
        "q": 1,
        "sparse": True,
    },
    loader_par={"batch_size": 128, "shuffle": True, "num_workers": 4},
    training_par={"optimizer_lr": 0.01, "num_epochs": 100},
):
    model = Node2Vec(data.edge_index, **node2vec_par).to(device)

    loader = model.loader(**loader_par)
    optimizer = torch.optim.SparseAdam(
        list(model.parameters()), lr=training_par["optimizer_lr"]
    )

    for epoch in range(1, training_par["num_epochs"] + 1):
        loss = train(model=model, optimizer=optimizer, loader=loader, device=device)
        # acc = test(model, data)
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")  # , Acc: {acc:.4f}')
    return model
