import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import signal
import os


class OrderTimeDomainAnalyzer:
    """
    A class for analyzing order-based time domain signals and creating comparison visualizations.
    """
    
    def __init__(self):
        """Initialize the OrderTimeDomainAnalyzer."""
        self.order_range = (0, 20)  # Order range from 0 to 20
        self.frequency_bands = 4    # Number of frequency bands
        self.file_count = 4         # Number of files to compare
        
    def create_comparison_visualization(self, data_dict, time_array, save_path=None):
        """
        Create a 4x4 comparison visualization of order spectrograms.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing data for 4 files and 4 frequency bands.
            Expected structure: {file_idx: {band_idx: spectrogram_data}}
        time_array : array-like
            Time array for the x-axis
        save_path : str, optional
            Path to save the image. If None, saves as 'order_spectrogram_comparison_1sec.png'
        
        Returns:
        --------
        str : Path where the image was saved
        """
        if save_path is None:
            save_path = 'order_spectrogram_comparison_1sec.png'
            
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        
        # Remove spacing between subplots for tight arrangement
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.95, 
                          wspace=0.02, hspace=0.02)
        
        # Generate order array for y-axis
        order_array = np.linspace(self.order_range[0], self.order_range[1], 100)
        
        # Global colorbar limits for consistent scaling
        vmin, vmax = -80, 20  # Typical dB range for spectrograms
        
        # Iterate through each subplot (4 files x 4 bands)
        for file_idx in range(self.file_count):
            for band_idx in range(self.frequency_bands):
                ax = axes[file_idx, band_idx]
                
                # Get or generate spectrogram data
                if data_dict and file_idx in data_dict and band_idx in data_dict[file_idx]:
                    spectrogram_data = data_dict[file_idx][band_idx]
                else:
                    # Generate sample data if not provided
                    spectrogram_data = self._generate_sample_spectrogram(
                        len(time_array), len(order_array), file_idx, band_idx
                    )
                
                # Ensure spectrogram_data has correct dimensions
                if spectrogram_data.shape != (len(order_array), len(time_array)):
                    # Resize if necessary
                    from scipy.interpolate import interp2d
                    f = interp2d(range(spectrogram_data.shape[1]), 
                               range(spectrogram_data.shape[0]), 
                               spectrogram_data, kind='linear')
                    spectrogram_data = f(np.linspace(0, spectrogram_data.shape[1]-1, len(time_array)),
                                       np.linspace(0, spectrogram_data.shape[0]-1, len(order_array)))
                
                # Convert to dB scale
                spectrogram_db = 20 * np.log10(np.abs(spectrogram_data) + 1e-10)
                
                # Create the spectrogram plot
                im = ax.pcolormesh(time_array, order_array, spectrogram_db, 
                                 shading='auto', cmap='viridis', 
                                 vmin=vmin, vmax=vmax)
                
                # Remove all labels and titles as requested
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('')
                
                # Set axis limits
                ax.set_xlim([time_array[0], time_array[-1]])
                ax.set_ylim([self.order_range[0], self.order_range[1]])
        
        # Add colorbar to the right side
        cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20)
        
        # Remove the overall figure title as requested
        fig.suptitle('')
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _generate_sample_spectrogram(self, time_len, order_len, file_idx, band_idx):
        """
        Generate sample spectrogram data for testing purposes.
        
        Parameters:
        -----------
        time_len : int
            Length of time dimension
        order_len : int
            Length of order dimension  
        file_idx : int
            File index (0-3)
        band_idx : int
            Band index (0-3)
            
        Returns:
        --------
        numpy.ndarray : Sample spectrogram data
        """
        # Create sample data with different patterns for each file/band combination
        np.random.seed(file_idx * 4 + band_idx)  # Consistent random seed
        
        # Base pattern with some variation based on file and band indices
        t = np.linspace(0, 1, time_len)
        orders = np.linspace(0, 20, order_len)
        
        # Create meshgrid
        T, O = np.meshgrid(t, orders)
        
        # Generate different patterns for different file/band combinations
        pattern1 = np.sin(2 * np.pi * (file_idx + 1) * T) * np.exp(-0.1 * O)
        pattern2 = np.cos(2 * np.pi * (band_idx + 1) * T) * np.exp(-0.05 * (O - 10)**2)
        noise = 0.1 * np.random.random((order_len, time_len))
        
        # Combine patterns
        spectrogram = np.abs(pattern1 + pattern2 + noise)
        
        # Add some frequency-specific features
        if band_idx == 0:  # Low frequency band
            spectrogram *= np.exp(-0.1 * O)
        elif band_idx == 1:  # Mid-low frequency band
            spectrogram *= np.exp(-0.05 * (O - 5)**2)
        elif band_idx == 2:  # Mid-high frequency band
            spectrogram *= np.exp(-0.05 * (O - 12)**2)
        else:  # High frequency band
            spectrogram *= np.exp(-0.1 * (20 - O))
            
        return spectrogram
    
    def load_data_from_files(self, file_paths, frequency_bands=None):
        """
        Load data from files (placeholder method - would need to be implemented
        based on actual data format).
        
        Parameters:
        -----------
        file_paths : list
            List of file paths to load data from
        frequency_bands : list, optional
            List of frequency band specifications
            
        Returns:
        --------
        dict : Dictionary containing loaded data
        """
        # This is a placeholder - actual implementation would depend on data format
        data_dict = {}
        for file_idx, file_path in enumerate(file_paths[:4]):
            data_dict[file_idx] = {}
            for band_idx in range(4):
                # Placeholder - would load actual data here
                data_dict[file_idx][band_idx] = None
                
        return data_dict


# Example usage and testing function
def test_order_time_domain_analyzer():
    """Test function for the OrderTimeDomainAnalyzer class."""
    analyzer = OrderTimeDomainAnalyzer()
    
    # Create sample time array (1 second duration)
    time_array = np.linspace(0, 1, 1000)
    
    # Create sample data (empty dict will use generated sample data)
    data_dict = {}
    
    # Create the visualization
    save_path = analyzer.create_comparison_visualization(data_dict, time_array)
    
    print(f"Visualization saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    test_order_time_domain_analyzer()