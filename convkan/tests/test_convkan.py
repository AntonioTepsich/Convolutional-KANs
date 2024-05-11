import pytest
import torch
from convkan import ConvKAN


@pytest.fixture
def sample_input():
    # Creates a sample input tensor to use in multiple tests
    return torch.randn(2, 3, 32, 32, dtype=torch.float32)


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,groups",
    [
        (3, 16, 3, 1, 1, 1),
        (3, 32, (3, 5), (1, 2), (1, 0), 1),
        (3, 18, 5, 1, 2, 1),
        (3, 18, 5, 1, 1, 3),  # Testing grouped convolution
    ],
)
def test_conv_kan_forward_shape(
    sample_input, in_channels, out_channels, kernel_size, stride, padding, groups
):
    model = ConvKAN(
        in_channels, out_channels, kernel_size, stride, padding, groups=groups
    )
    output = model(sample_input)
    expected_height = (
        sample_input.size(2) + 2 * model.padding[0] - model.kernel_size[0]
    ) // model.stride[0] + 1
    expected_width = (
        sample_input.size(3) + 2 * model.padding[1] - model.kernel_size[1]
    ) // model.stride[1] + 1
    assert output.shape == (
        2,
        out_channels,
        expected_height,
        expected_width,
    ), "Output shape is incorrect."


def test_invalid_groups():
    with pytest.raises(ValueError):
        # in_channels not divisible by groups
        ConvKAN(3, 16, 3, groups=2)


@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
def test_padding_modes(sample_input, padding_mode):
    kernel_size = 3
    stride = 1
    padding = 1
    in_channels = 3
    out_channels = 16

    model = ConvKAN(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
    )
    output = model(sample_input)

    # Calculating expected dimensions
    input_height, input_width = sample_input.shape[2:4]
    expected_height = (input_height + 2 * padding - kernel_size) // stride + 1
    expected_width = (input_width + 2 * padding - kernel_size) // stride + 1

    assert output.shape == (
        sample_input.shape[0],
        out_channels,
        expected_height,
        expected_width,
    ), (
        f"Output shape is incorrect for padding mode {padding_mode}. "
        f"Expected ({sample_input.shape[0], out_channels, expected_height, expected_width}), got {output.shape}"
    )


@pytest.mark.parametrize(
    "in_channels, out_channels, groups, kernel_size",
    [(4, 8, 2, (3, 3)), (6, 12, 3, (3, 3))],
)
def test_group_effect_on_output(in_channels, out_channels, groups, kernel_size):
    # Initialize the model
    model = ConvKAN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
    )

    # Create base input tensor
    base_input = torch.rand(1, in_channels, 10, 10)

    # Create a modified input tensor by altering the channels of one specific group
    modified_input = base_input.clone()
    group_size = in_channels // groups
    group_to_modify = 1  # Let's modify the second group (0-indexed)
    modified_input[
        :, group_size * group_to_modify : group_size * (group_to_modify + 1)
    ] *= 2  # Scale the group's channels

    # Get outputs from the model
    output_base = model(base_input)
    output_modified = model(modified_input)

    # Check that outputs for the unchanged groups remain the same
    for g in range(groups):
        start_channel = g * (out_channels // groups)
        end_channel = start_channel + (out_channels // groups)

        if g == group_to_modify:
            assert not torch.allclose(
                output_base[:, start_channel:end_channel],
                output_modified[:, start_channel:end_channel],
                atol=1e-6,
            ), f"Output channels {start_channel}-{end_channel} should be different due to input modification."
        else:
            assert torch.allclose(
                output_base[:, start_channel:end_channel],
                output_modified[:, start_channel:end_channel],
                atol=1e-6,
            ), f"Output channels {start_channel}-{end_channel} should be unchanged."


def test_gradient_flow(sample_input):
    model = ConvKAN(3, 16, 3, stride=1, padding=1)
    model.train()
    output = model(sample_input)
    loss = output.sum()
    loss.backward()
    # Check if gradients are None or not (simple gradient existence check)
    for param in model.parameters():
        assert param.grad is not None, "Gradients should not be None."


def test_dtype_handling():
    model = ConvKAN(3, 16, 3)
    inputs = torch.randn(2, 3, 32, 32, dtype=torch.double)
    model.double()  # Set model to double precision
    output = model(inputs)
    assert (
        output.dtype == torch.double
    ), "Output dtype should match input dtype (double)."
