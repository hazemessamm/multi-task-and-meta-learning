"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing
        
        self.users_embedding = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.items_embedding = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        if not embedding_sharing:
            self.users_embedding_mlp = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
            self.items_embedding_mlp = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.A = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.B = ZeroEmbedding(num_items, 1, sparse=sparse)


        self.mlp = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.mlp.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.mlp.append(nn.ReLU())
        
        self.mlp.append(nn.Linear(layer_sizes[-1], 1))
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """

        users_embed = self.users_embedding(user_ids)
        items_embed = self.items_embedding(item_ids)

        users_bias = self.A(user_ids)
        items_bias = self.B(item_ids)
        

        o = users_embed * items_embed
        o = torch.sum(o, dim=-1)
        predictions = o.unsqueeze(-1) + users_bias + items_bias
        predictions = predictions.squeeze(-1)
        
        elem_wise_mult = users_embed * items_embed

        if self.embedding_sharing:
            out = torch.cat([users_embed, items_embed, elem_wise_mult], dim=-1)
        else:
            users_embed_mlp = self.users_embedding_mlp(user_ids)
            items_embed_mlp = self.items_embedding_mlp(item_ids)
            out = torch.cat([users_embed_mlp, items_embed_mlp, elem_wise_mult])

        for layer in self.mlp:
            out = layer(out)

        score = out.squeeze(-1)
        return predictions, score