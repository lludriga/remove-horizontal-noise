import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

class VideoPCA:
    def __init__(self, video_files):
        self.video_files = video_files
        self.data = None
        self.pca = None
        self.cumulative_variance = None
        self.frame_shape = None

    def flatten_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_shape is None:
                self.frame_shape = (frame.shape[0], frame.shape[1])  # Set the frame shape
                print(f"Determined frame shape: {self.frame_shape}")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flat_frame = gray_frame.flatten()
            frames.append(flat_frame)

        cap.release()
        return np.array(frames)

    def collect_frames(self):
        all_frames = []
        for video_file in self.video_files:
            frames = self.flatten_frames(video_file)
            all_frames.append(frames)
        self.data = np.vstack(all_frames)
        print(f"Collected data shape: {self.data.shape}")

    def pca_videos(self):
        if self.data is None:
            raise ValueError("Data not collected. Please run collect_frames() first.")

        self.pca = PCA()
        self.pca.fit(self.data)
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

    def plot_cumulative_variance(self, output_dir):
        if self.cumulative_variance is None:
            raise ValueError("PCA not performed. Please run pca_videos() first.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Variance Explained by PCA Components')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'cumulative_variance.png'))
        plt.close()

    def get_n_components_for_variance(self, threshold=0.80):
        if self.cumulative_variance is None:
            raise ValueError("PCA not performed. Please run pca_videos() first.")
        return np.argmax(self.cumulative_variance >= threshold) + 1

class VideoICA:
    def __init__(self, data, frame_shape, video_files):
        self.data = data
        self.frame_shape = frame_shape
        self.video_files = video_files
        self.ica = None
        self.sources = None

    def perform_ica(self, n_components, max_iter=1000, tol=1e-4):
        if self.data is None:
            raise ValueError("Data not collected. Please provide the data.")
        self.ica = FastICA(n_components=n_components, random_state=0, max_iter=max_iter, tol=tol)
        self.sources = self.ica.fit_transform(self.data)
        print(f"ICA performed, sources shape: {self.sources.shape}")

    def plot_ica_components_as_images(self, output_dir='.', components_per_fig=100):
        if self.sources is None:
            raise ValueError("ICA not performed. Please run perform_ica() first.")

        os.makedirs(output_dir, exist_ok=True)
        num_components = self.sources.shape[1]

        n_cols = 10
        n_rows = min(components_per_fig // n_cols, num_components)

        for fig_num in range(0, num_components, components_per_fig):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2))
            for i in range(components_per_fig):
                comp_idx = fig_num + i
                if comp_idx >= num_components:
                    break
                ax = axes[i // n_cols, i % n_cols]
                try:
                    component_image = self.ica.components_[comp_idx].reshape(self.frame_shape)
                except ValueError as e:
                    print(f"Error reshaping component {comp_idx}: {e}")
                    continue
                ax.imshow(component_image, cmap='jet', aspect='auto')
                ax.set_title(f'Component {comp_idx + 1}')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ica_components_images_{fig_num // components_per_fig}.png'))
            plt.close()

    def identify_noise_components(self, num_components=5):
        component_variances = np.var(self.sources, axis=0)
        print(f"Component variances: {component_variances}")
        noise_components = np.argsort(component_variances)[-num_components:]
        print(f"Top {num_components} noise components by variance: {noise_components}")
        return noise_components.tolist()

    def remove_noise_and_reconstruct(self, noise_components):
        if self.sources is None:
            raise ValueError("ICA not performed. Please run perform_ica() first.")
        cleaned_sources = self.sources.copy()
        cleaned_sources[:, noise_components] = 0
        reconstructed_data = self.ica.inverse_transform(cleaned_sources)
        return reconstructed_data

    def sum_non_discarded_and_reconstruct(self, noise_components):
        if self.sources is None:
            raise ValueError("ICA not performed. Please run perform_ica() first.")
        # Identify non-discarded components
        all_components = np.arange(self.sources.shape[1])
        non_discarded_components = np.setdiff1d(all_components, noise_components)
        print(f"Non-discarded components: {non_discarded_components}")

        # Sum non-discarded components to reconstruct the data
        reconstructed_data = self.ica.mixing_[:, non_discarded_components] @ self.sources[:, non_discarded_components].T
        reconstructed_data = reconstructed_data.T + self.ica.mean_

        return reconstructed_data

    def save_reconstructed_videos(self, cleaned_data, output_dir, suffix='_ica'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate the number of frames per video
        frames_per_video = len(cleaned_data) // len(self.video_files)
        
        for idx, video_file in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {video_file}")
            
            # Get the original frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Define the codec and create a VideoWriter object
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}{suffix}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, fps, (self.frame_shape[1], self.frame_shape[0]), isColor=False)
            
            for i in range(frames_per_video):
                frame_index = idx * frames_per_video + i
                frame_data = cleaned_data[frame_index].reshape(self.frame_shape)
                frame_data = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                out.write(frame_data)
            
            cap.release()
            out.release()
            print(f"Saved reconstructed video to {output_file}")

    def save_ica_components(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'ica_components.npy'), self.ica.components_)
        np.save(os.path.join(output_dir, 'ica_sources.npy'), self.sources)
        np.save(os.path.join(output_dir, 'ica_mixing.npy'), self.ica.mixing_)
        np.save(os.path.join(output_dir, 'ica_mean.npy'), self.ica.mean_)
        print(f"ICA components, sources, mixing matrix, and mean saved to {output_dir}")

    def load_ica_components(self, output_dir):
        if self.ica is None:
            self.ica = FastICA()  # Initialize the ICA object if not already done
        self.ica.components_ = np.load(os.path.join(output_dir, 'ica_components.npy'))
        self.sources = np.load(os.path.join(output_dir, 'ica_sources.npy'))
        self.ica.mixing_ = np.load(os.path.join(output_dir, 'ica_mixing.npy'))
        self.ica.mean_ = np.load(os.path.join(output_dir, 'ica_mean.npy'))
        print(f"ICA components, sources, mixing matrix, and mean loaded from {output_dir}")

# Example usage
def run_analysis():
    dir = '.'
    avi_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.avi') and not f.startswith('p_')]
    avi_files.sort()
    print(avi_files)

    output_dir = './output_plots'
    os.makedirs(output_dir, exist_ok=True)

    video_pca = VideoPCA(avi_files)
    video_pca.collect_frames()
    print("collected frames successfully")
    video_pca.pca_videos()
    print("pca performed")
    video_pca.plot_cumulative_variance(output_dir=output_dir)
    print("plot saved")

    n_components = video_pca.get_n_components_for_variance(threshold=0.80)
    print(f"Number of components explaining 80% of the variance: {n_components}")

    #frame_shape = video_pca.frame_shape
    #print(f"Frame shape for ICA: {frame_shape}")
    frame_shape = (606, 542)
    video_ica = VideoICA(video_pca.data, frame_shape=frame_shape, video_files=avi_files)
    video_ica.perform_ica(n_components=100, max_iter=1000, tol=1e-4)
    print("ica performed")
    video_ica.plot_ica_components_as_images(output_dir=output_dir, components_per_fig=100)
    print("ica components plotted and saved")
    video_ica.save_ica_components(output_dir)
    print("Ica components correctly saved")


    # Identified noise components from the images
    noise_components = [3, 6, 15, 23, 32, 35, 38, 42, 43, 44, 45, 51, 58, 61]
    print(f"Identified noise components: {noise_components}")

    cleaned_data = video_ica.remove_noise_and_reconstruct(noise_components)
    print("Noise components removed and data reconstructed")

    video_ica.save_reconstructed_videos(cleaned_data, output_dir, suffix='_cleaned')
    print("Reconstructed videos saved")

    # Reconstruct the video using the sum of non-discarded components
    sum_non_discarded_data = video_ica.sum_non_discarded_and_reconstruct(noise_components)
    video_ica.save_reconstructed_videos(sum_non_discarded_data, output_dir, suffix='_sum_non_discarded')
    print("Videos reconstructed from non-discarded components saved")

    video_ica.save_ica_components(output_dir)
    print("ICA components and sources saved")

# Run the analysis
run_analysis()
