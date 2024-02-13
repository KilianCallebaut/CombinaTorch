import torch

from Training.Results import Results


class TrainingUtils:

    def loss_calculations(self, task_list, batch_flags, target_flags, output, labels, device):
        n_tasks = len(task_list)
        losses_batch = [torch.tensor([0]).to(device) for _ in task_list]
        output_batch = [torch.Tensor([]) for _ in task_list]
        labels_batch = [torch.Tensor([]) for _ in task_list]
        for i in range(n_tasks):
            if sum(batch_flags[i]) == 0:
                continue

            filtered_output = output[i]
            filtered_labels = labels

            if n_tasks > 1:
                filtered_output = filtered_output[batch_flags[i], :]

                filtered_labels = filtered_labels[batch_flags[i], :]
                filtered_labels = filtered_labels[:, target_flags[i]]
            filtered_labels = task_list[i].translate_labels(filtered_labels)

            losses_batch[i] = task_list[i].loss_function(filtered_output, filtered_labels)
            output_batch[i] = task_list[i].decision_making(filtered_output).detach()
            labels_batch[i] = filtered_labels.detach()
        return losses_batch, output_batch, labels_batch

    def combine_loss(self, list_of_losses):
        return sum([l for l in list_of_losses])

    def early_stop(self, results: Results, epoch: int):
        if epoch == 0:
            return False
        if (results.training_curve[epoch - 1] - results.training_curve[epoch]) < 0.0001:
            return True

        return False

    def extra_operation(self, **kwargs):
        pass
