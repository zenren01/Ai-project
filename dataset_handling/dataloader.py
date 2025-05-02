class CustomDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Shuffle the dataset indices if needed
        indices = torch.randperm(len(self.dataset))

        # Generate mini-batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]

            # Yield the batch data
            yield batch_data

    def __len__(self):
        # Calculate the number of mini-batches
        return len(self.dataset) // self.batch_size

batch_size = 1
##  
train_loader_m = CustomDataLoader(custom_dataset_m, batch_size)


# OR JUST USING pytorch
# train_loader_m =torch.utils.data.DataLoader(custom_dataset_m, batch_size = 1)


# Obtain a batch of data
##
blur, sharp = next(iter(train_loader_m))