#!/usr/bin/env python3
"""
Convert your trained model to match the new CLAP structure
"""

import torch
import os

def convert_clap_structure(ckpt_path, output_path):
    """Convert direct CLAP structure to sequential structure"""
    print(f"🔄 Converting {ckpt_path}")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Convert both state_dict and ema if they exist
    for key_name in ['state_dict', 'ema']:
        if key_name in ckpt:
            weights = ckpt[key_name]
            converted_weights = {}
            
            for weight_key, weight_value in weights.items():
                new_key = weight_key
                
                # Convert CLAP transform keys
                if 'clap.model.text_transform.' in weight_key and '.sequential.' not in weight_key:
                    # clap.model.text_transform.0.weight -> clap.model.text_transform.sequential.0.weight
                    new_key = weight_key.replace('clap.model.text_transform.', 'clap.model.text_transform.sequential.')
                    print(f"  Converting: {weight_key} -> {new_key}")
                    
                elif 'clap.model.audio_transform.' in weight_key and '.sequential.' not in weight_key:
                    # clap.model.audio_transform.0.weight -> clap.model.audio_transform.sequential.0.weight  
                    new_key = weight_key.replace('clap.model.audio_transform.', 'clap.model.audio_transform.sequential.')
                    print(f"  Converting: {weight_key} -> {new_key}")
                
                converted_weights[new_key] = weight_value
            
            ckpt[key_name] = converted_weights
            print(f"✅ Converted {len(weights)} keys in {key_name}")
    
    # Save converted checkpoint
    torch.save(ckpt, output_path)
    print(f"💾 Saved converted model to: {output_path}")

def main():
    # Convert your trained model
    original_path = "/home/husrcf/Code/Python/ProjectLily_Z_III/outputs/audiosr_ldm_train/checkpoints/step_30000.pt"
    converted_path = "/home/husrcf/Code/Python/ProjectLily_Z_III/outputs/audiosr_ldm_train/checkpoints/step_30000_converted.pt"
    
    if os.path.exists(original_path):
        convert_clap_structure(original_path, converted_path)
        
        # Verify the conversion
        print("\n🔍 Verifying conversion...")
        ckpt = torch.load(converted_path, map_location='cpu')
        
        if 'state_dict' in ckpt:
            clap_keys = [k for k in ckpt['state_dict'].keys() if 'clap.model' in k and 'transform' in k]
        else:
            clap_keys = [k for k in ckpt['ema'].keys() if 'clap.model' in k and 'transform' in k]
            
        print("✅ CLAP keys after conversion:")
        for key in clap_keys:
            print(f"  {key}")
            
        print(f"\n🎉 Conversion complete! Use {converted_path} with your updated model.")
    else:
        print(f"❌ Original model not found: {original_path}")

if __name__ == "__main__":
    main()