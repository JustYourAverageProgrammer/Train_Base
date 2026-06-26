from torch import nn

DROPOUTS = {

    "Dropout"       : nn.Dropout,
    "Dropout2d"     : nn.Dropout2d,
    "AlphaDropout"  : nn.AlphaDropout,
}

def get_dropout(model_config):
    dropout_config = model_config["dropout"]
    dropout_name   = dropout_config["type"]
    params         = {k: v for k, v in dropout_config.items() if k != "type" and v is not None}
    return DROPOUTS[dropout_name](**params)


# Single Convolutional Block - two conv layers that double the channel depth while preserving spatial size
class argus_conv_block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),   # Preserves spatial size
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.1),                                              # Leaky ReLU prevents dead neurons
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),  # Preserves spatial size
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.block(x)


class argus(nn.Module):

    def __init__(self, model_config, dataset_info):
        super().__init__()

        num_classes = dataset_info.num_classes   # numclasses of dataset
        in_channels = dataset_info.num_channels  # initial in channels = numclasses

        out_channels = 32       # first layer of out classes

        blocks = [argus_conv_block(in_channels, out_channels)]  # First block 3 - 32
        in_channels  = out_channels                             # Update inchannels = outchannels
        out_channels = in_channels * 2                          # Multiply ouchannels for next one * 2

        for _ in range(3):
            blocks.append(argus_conv_block(in_channels, out_channels))  # Stack three leaky residual conneciton blocks
            in_channels  = out_channels                                 # Update channels in and out
            out_channels = in_channels * 2                              # Update channels in and out

        self.blocks  = nn.ModuleList(blocks)
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.dropout = get_dropout(model_config)
        self.nin     = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.blocks[0](x)              # 3→32,   32×32  (no pool)
        for block in self.blocks[1:]:
            x = self.maxPool(block(x))     # 32→64 16×16, 64→128 8×8, 128→256 4×4

        x = self.dropout(x)

        x = self.nin(x)
        x = self.gap(x)

        return self.flatten(x)

# Models can be added to the registry as created/needed
MODELS = {
    "argus"     :   argus,
}
