import torch
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call


def compute_jvp(function, input_tensor, vectors):
    """
    Compute the Jacobian-vector product for a given function and input tensor.

    Parameters:
    - function: A differentiable function that takes a tensor and returns a tensor.
    - input_tensor: A tensor which is the input to the function.
    - vector: A vector (tensor) to backpropagate.

    Returns:
    - The Jacobian-vector product as a tensor.
    """
    # Assign the tangent to each weight
    params = {
        name: weight
        for name, weight in function.named_parameters()
        if weight.requires_grad
    }
    dual_params = {}
    with fwAD.dual_level():
        for idx, (name, p) in enumerate(params.items()):
            dual_params[name] = fwAD.make_dual(p, vectors[idx])
        # perform forward pass with dual weights
        out = functional_call(function, dual_params, input_tensor)
        # extract JVP
        jvp = fwAD.unpack_dual(out).tangent

    return jvp


def compute_vjp(function, input_tensor, vector):
    """
    Compute the vector-Jacobian product for a given function and input tensor.

    Parameters:
    - function: A differentiable function that takes a tensor and returns a tensor.
    - input_tensor: A tensor which is the input to the function.
    - vector: A vector (tensor) to backpropagate.

    Returns:
    - The vector-Jacobian product as a tensor.
    """
    # Ensure the input tensor requires gradient computation
    input_tensor.requires_grad_(True)

    # Forward pass: compute the function output
    output = function(input_tensor)

    # Check if the vector has the same size as the output
    if output.size() != vector.size():
        raise ValueError("The vector must have the same size as the function output.")

    # compute VJP
    # TODO: autograd.grad has an argument is_grads_batched which could be useful
    weights = [w for w in function.parameters() if w.requires_grad]
    vjp = torch.autograd.grad(
        outputs=output,
        inputs=weights,
        grad_outputs=vector,
        # is_grads_batched=True,
        retain_graph=False,
        allow_unused=False,
    )

    return vjp


def compute_ntk_similarity(model1, model2, input_tensor, mode="ntk", num_samples=100):
    """
    Compute the similarity between the Neural Tangent Kernels (NTKs) of two models.

    Parameters:
    - model1, model2: Two instances of a PyTorch model.
    - input_tensor: A tensor which is the input to the models.
    - mode: specifies which similarity metric to use, either "jacobian" or "ntk"

    Returns:
    - The similarity metric between the two NTKs.
    """

    # Define a function to compute the VJP for a model
    def model_vjps(model1, model2, input_tensor):
        # Use a random vector for the VJP computation
        vector = torch.randn(*model1(input_tensor).size()).to(input_tensor.device)
        return compute_vjp(model1, input_tensor, vector), compute_vjp(
            model2, input_tensor, vector
        )

    # Compute VJPs for both models
    all_values1, all_values2 = [], []
    for _ in range(num_samples):
        vjp1, vjp2 = model_vjps(model1, model2, input_tensor)

        if mode == "jacobian":
            vjp1 = torch.cat([v.flatten() for v in vjp1])
            vjp2 = torch.cat([v.flatten() for v in vjp2])
            all_values1.append(vjp1.flatten()); all_values2.append(vjp2.flatten())
            # similarity = torch.dot(vjp1.flatten(), vjp2.flatten()) / (
                # torch.dot(vjp1.flatten(), vjp1.flatten()) ** 0.5
                # * torch.dot(vjp2.flatten(), vjp2.flatten()) ** 0.5
            # )
        else:
            jvp1 = compute_jvp(model1, input_tensor, vjp1)
            jvp2 = compute_jvp(model2, input_tensor, vjp2)
            all_values1.append(jvp1.flatten()); all_values2.append(jvp2.flatten())
            # similarity = torch.dot(jvp1.flatten(), jvp2.flatten()) / (
                # torch.dot(jvp1.flatten(), jvp1.flatten()) ** 0.5
                # * torch.dot(jvp2.flatten(), jvp2.flatten()) ** 0.5
            # )

    all_values1 = torch.mean(torch.stack(all_values1), dim=0)
    all_values2 = torch.mean(torch.stack(all_values2), dim=0)

    similarity = torch.dot(all_values1.flatten(), all_values2.flatten()) / (
        torch.dot(all_values1.flatten(), all_values1.flatten()) ** 0.5
        * torch.dot(all_values2.flatten(), all_values2.flatten()) ** 0.5
    )


    return similarity


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torch import nn, optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 784  # 28x28 images flattened
    num_classes = 10  # 10 classes for MNIST digits
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5
    non_linearity = True
    mode = "ntk"

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="../../research/data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="../../research/data", train=False, transform=transforms.ToTensor()
    )

    # Data loader
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    if non_linearity:
        model = nn.Sequential(nn.Linear(input_size, num_classes), nn.ReLU()).to(device)
        initial_model = nn.Sequential(nn.Linear(input_size, num_classes), nn.ReLU()).to(device)
    else:
        model = nn.Linear(input_size, num_classes).to(device)
        initial_model = nn.Linear(input_size, num_classes).to(device)
    initial_model.load_state_dict(model.state_dict())

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Flatten the images
            images = images.view(-1, 28 * 28).to(device)

            # Forward pass
            outputs = model(images)
            # Convert labels to one-hot encoding
            labels_one_hot = (
                torch.nn.functional.one_hot(labels, num_classes).float().to(device)
            )
            loss = criterion(outputs, labels_one_hot)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        # Compute and print the NTK similarity with the initial model
        ntk_similarity = compute_ntk_similarity(
            initial_model, model, images, mode=mode
        )
        print(f"Epoch {epoch}, NTK Similarity with Initial: {ntk_similarity}")

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

        print(
            f"Accuracy of the model on the 10000 test images: {100 * correct / total}%"
        )
