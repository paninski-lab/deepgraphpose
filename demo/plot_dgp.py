from deepgraphpose.models.eval import plot_dgp

if __name__ == '__main__':
    plot_dgp(video_file=str(video_file_name),
             output_dir=output_dir,
             proj_cfg_file=str(cfg_yaml),
             dgp_model_file=str(snapshot_path),
             shuffle=shuffle)