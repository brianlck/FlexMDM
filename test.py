from model.transformer import SemiAutoregressiveFlow
import torch



if __name__ == '__main__':
    from omegaconf import OmegaConf
    # Create a simple config
    config = OmegaConf.create({
        'tokens': 256,
        'model': {
            'hidden_size': 512,
            'n_heads': 8,
            'cond_dim': 128,
            'dropout': 0.1,
            'n_blocks': 4,
            'scale_by_sigma': True
        }
    })

    # Initialize model
    model = SemiAutoregressiveFlow(config)
    model = model.cuda()

    # Create sample inputs
    batch_size = 2
    seq_len = 16
    indices = torch.randint(0, config.tokens, (batch_size, seq_len)).cuda()
    sigma = torch.randn(batch_size).cuda()

    # Do forward pass
    unmask_rate, len_rate = model(indices, sigma)

    # Print shapes
    print(f"Input shapes:")
    print(f"indices: {indices.shape}")
    print(f"sigma: {sigma.shape}")
    print(f"\nOutput shapes:")
    print(f"unmask_rate: {unmask_rate.shape}")
    print(f"len_rate: {len_rate.shape}")

    # Verify outputs
    assert unmask_rate.shape == (batch_size, seq_len, config.tokens)
    assert len_rate.shape == (batch_size,)
    print("\nForward pass successful!")