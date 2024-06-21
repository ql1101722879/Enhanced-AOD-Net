import torchinfo

import net

torchinfo.summary(net.AODnet(), (3, 640, 480), batch_dim=0,
                      col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=0)