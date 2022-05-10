import torch

def dense_to_sparse_3d(dense):
    print()


if __name__ == "__main__":
    # Dense tensor to be converted to sparse
    dense = torch.Tensor([
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ],
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ],
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ]
    ]).cuda()

    print(dense.size())
    print(dense)
    
    #values, row_indices, offsets, column_indices = dense_to_sparse_3d(dense)