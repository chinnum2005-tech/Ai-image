"""
Pixel-wise Feature Extraction Module for AI-Generated Image Detection
Implements PRNU, ELA, inter-pixel correlation, and texture analysis
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy import signal, ndimage
from scipy.fftpack import dct, idct
import warnings
warnings.filterwarnings('ignore')


class PixelWiseFeatureExtractor:
    """
    Extracts pixel-level features to detect AI-generated images
    """
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, image_path):
        """
        Extract comprehensive pixel-wise features from an image
        """
        # Load image
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. PRNU (Photo Response Non-Uniformity) Analysis
        prnu_features = self.extract_prnu_features(img_gray)
        features.update(prnu_features)
        
        # 2. Error Level Analysis (ELA)
        ela_features = self.extract_ela_features(image_path)
        features.update(ela_features)
        
        # 3. Inter-pixel Correlation Analysis
        correlation_features = self.extract_correlation_features(img_gray)
        features.update(correlation_features)
        
        # 4. Residual Map Analysis
        residual_features = self.extract_residual_features(img_gray)
        features.update(residual_features)
        
        # 5. Texture and Noise Features
        texture_features = self.extract_texture_features(img_gray)
        features.update(texture_features)
        
        # 6. Color Space Analysis
        color_features = self.extract_color_features(img_rgb)
        features.update(color_features)
        
        # 7. Frequency Domain Features
        freq_features = self.extract_frequency_features(img_gray)
        features.update(freq_features)
        
        return features
    
    def extract_prnu_features(self, img_gray):
        """
        Extract Photo Response Non-Uniformity features
        Real cameras have unique sensor noise patterns
        """
        features = {}
        
        # Denoise image using wavelet denoising
        denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)
        
        # Calculate noise residual
        noise_residual = img_gray.astype(float) - denoised.astype(float)
        
        # Apply high-pass filter to extract sensor pattern noise
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        high_freq_noise = cv2.filter2D(noise_residual, -1, kernel)
        
        # Statistical features of PRNU
        features['prnu_mean'] = np.mean(high_freq_noise)
        features['prnu_std'] = np.std(high_freq_noise)
        features['prnu_skew'] = self.calculate_skewness(high_freq_noise)
        features['prnu_kurtosis'] = self.calculate_kurtosis(high_freq_noise)
        
        # Local variance of noise
        local_var = ndimage.generic_filter(high_freq_noise, np.var, size=5)
        features['prnu_local_var_mean'] = np.mean(local_var)
        features['prnu_local_var_std'] = np.std(local_var)
        
        # Noise consistency (real cameras have consistent noise patterns)
        noise_blocks = self.split_into_blocks(high_freq_noise, 32)
        block_vars = [np.var(block) for block in noise_blocks]
        features['prnu_consistency'] = np.std(block_vars) / (np.mean(block_vars) + 1e-10)
        
        return features
    
    def extract_ela_features(self, image_path):
        """
        Error Level Analysis - detects compression artifacts
        AI images often have different compression patterns
        """
        features = {}
        
        # Open original image
        img = Image.open(image_path).convert('RGB')
        
        # Save at lower quality and reload
        img_compressed = img.copy()
        img_compressed.save('temp_compressed.jpg', quality=90)
        compressed = Image.open('temp_compressed.jpg')
        
        # Calculate difference
        ela_img = ImageChops.difference(img, compressed)
        
        # Enhance difference for analysis
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff + 1e-10)
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        
        # Convert to numpy array
        ela_array = np.array(ela_img)
        
        # Extract features from ELA
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = ela_array[:, :, i]
            features[f'ela_{channel}_mean'] = np.mean(channel_data)
            features[f'ela_{channel}_std'] = np.std(channel_data)
            features[f'ela_{channel}_max'] = np.max(channel_data)
            
        # Overall ELA statistics
        ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
        features['ela_energy'] = np.sum(ela_gray ** 2) / ela_gray.size
        features['ela_entropy'] = self.calculate_entropy(ela_gray)
        
        # Clean up temp file
        import os
        if os.path.exists('temp_compressed.jpg'):
            os.remove('temp_compressed.jpg')
        
        return features
    
    def extract_correlation_features(self, img_gray):
        """
        Inter-pixel correlation analysis
        AI-generated images often have different correlation patterns
        """
        features = {}
        
        # Horizontal correlation
        h_corr = np.corrcoef(img_gray[:, :-1].flatten(), 
                            img_gray[:, 1:].flatten())[0, 1]
        features['h_correlation'] = h_corr
        
        # Vertical correlation
        v_corr = np.corrcoef(img_gray[:-1, :].flatten(), 
                            img_gray[1:, :].flatten())[0, 1]
        features['v_correlation'] = v_corr
        
        # Diagonal correlation
        d_corr = np.corrcoef(img_gray[:-1, :-1].flatten(), 
                            img_gray[1:, 1:].flatten())[0, 1]
        features['d_correlation'] = d_corr
        
        # Anti-diagonal correlation
        ad_corr = np.corrcoef(img_gray[:-1, 1:].flatten(), 
                             img_gray[1:, :-1].flatten())[0, 1]
        features['ad_correlation'] = ad_corr
        
        # Correlation variance (consistency measure)
        features['correlation_variance'] = np.var([h_corr, v_corr, d_corr, ad_corr])
        
        # Local correlation patterns
        local_correlations = []
        for i in range(0, img_gray.shape[0]-16, 16):
            for j in range(0, img_gray.shape[1]-16, 16):
                block = img_gray[i:i+16, j:j+16]
                if block.size > 0:
                    h_local = np.corrcoef(block[:, :-1].flatten(), 
                                         block[:, 1:].flatten())[0, 1]
                    if not np.isnan(h_local):
                        local_correlations.append(h_local)
        
        if local_correlations:
            features['local_corr_mean'] = np.mean(local_correlations)
            features['local_corr_std'] = np.std(local_correlations)
        else:
            features['local_corr_mean'] = 0
            features['local_corr_std'] = 0
        
        return features
    
    def extract_residual_features(self, img_gray):
        """
        Residual map computation for detecting synthesis artifacts
        """
        features = {}
        
        # Apply various filters to get residuals
        # Gaussian residual
        gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
        gaussian_residual = img_gray.astype(float) - gaussian.astype(float)
        
        features['gaussian_res_mean'] = np.mean(gaussian_residual)
        features['gaussian_res_std'] = np.std(gaussian_residual)
        features['gaussian_res_energy'] = np.sum(gaussian_residual ** 2) / gaussian_residual.size
        
        # Median filter residual (good for detecting synthesis artifacts)
        median = cv2.medianBlur(img_gray, 5)
        median_residual = img_gray.astype(float) - median.astype(float)
        
        features['median_res_mean'] = np.mean(median_residual)
        features['median_res_std'] = np.std(median_residual)
        features['median_res_max'] = np.max(np.abs(median_residual))
        
        # Bilateral filter residual
        bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
        bilateral_residual = img_gray.astype(float) - bilateral.astype(float)
        
        features['bilateral_res_mean'] = np.mean(bilateral_residual)
        features['bilateral_res_std'] = np.std(bilateral_residual)
        
        # Residual pattern regularity (AI images may have regular patterns)
        residual_fft = np.fft.fft2(gaussian_residual)
        residual_spectrum = np.abs(residual_fft)
        features['residual_spectral_energy'] = np.sum(residual_spectrum) / residual_spectrum.size
        
        return features
    
    def extract_texture_features(self, img_gray):
        """
        Extract texture and noise features using various methods
        """
        features = {}
        
        # Local Binary Patterns (LBP) for texture
        lbp = self.calculate_lbp(img_gray)
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_entropy'] = self.calculate_entropy(lbp.astype(np.uint8))
        
        # Gabor filter responses
        gabor_features = self.apply_gabor_filters(img_gray)
        features.update(gabor_features)
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        glcm_features = self.calculate_glcm_features(img_gray)
        features.update(glcm_features)
        
        # Noise statistics
        noise = self.estimate_noise(img_gray)
        features['noise_level'] = noise
        
        # Edge density (AI images may have different edge characteristics)
        edges = cv2.Canny(img_gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Texture homogeneity
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        features['texture_variance'] = np.var(laplacian)
        
        return features
    
    def extract_color_features(self, img_rgb):
        """
        Color space and quantization analysis
        """
        features = {}
        
        # Convert to different color spaces
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
        # Color distribution statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            data = img_rgb[:, :, i]
            features[f'{channel}_mean'] = np.mean(data)
            features[f'{channel}_std'] = np.std(data)
            features[f'{channel}_skew'] = self.calculate_skewness(data)
            
        # HSV statistics (AI images may have different saturation patterns)
        features['hue_mean'] = np.mean(img_hsv[:, :, 0])
        features['saturation_mean'] = np.mean(img_hsv[:, :, 1])
        features['saturation_std'] = np.std(img_hsv[:, :, 1])
        features['value_mean'] = np.mean(img_hsv[:, :, 2])
        
        # LAB color space features
        features['lab_l_std'] = np.std(img_lab[:, :, 0])
        features['lab_a_range'] = np.ptp(img_lab[:, :, 1])
        features['lab_b_range'] = np.ptp(img_lab[:, :, 2])
        
        # Color quantization consistency
        quantized = (img_rgb // 32) * 32  # 3-bit quantization
        quant_diff = img_rgb.astype(float) - quantized.astype(float)
        features['quantization_error'] = np.mean(np.abs(quant_diff))
        
        # Color diversity
        unique_colors = len(np.unique(img_rgb.reshape(-1, 3), axis=0))
        features['color_diversity'] = unique_colors / (img_rgb.shape[0] * img_rgb.shape[1])
        
        return features
    
    def extract_frequency_features(self, img_gray):
        """
        Frequency domain analysis using FFT and DCT
        """
        features = {}
        
        # FFT analysis
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Get center for radial analysis
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Low frequency energy (center region)
        r = 30
        mask_low = np.zeros((rows, cols))
        cv2.circle(mask_low, (ccol, crow), r, 1, -1)
        low_freq_energy = np.sum(magnitude_spectrum * mask_low) / np.sum(mask_low)
        
        # High frequency energy (outer region)
        mask_high = 1 - mask_low
        high_freq_energy = np.sum(magnitude_spectrum * mask_high) / np.sum(mask_high)
        
        features['low_freq_energy'] = low_freq_energy
        features['high_freq_energy'] = high_freq_energy
        features['freq_ratio'] = low_freq_energy / (high_freq_energy + 1e-10)
        
        # DCT analysis
        dct_coeffs = dct(dct(img_gray.T, norm='ortho').T, norm='ortho')
        
        # Energy in different frequency bands
        features['dct_low_energy'] = np.sum(np.abs(dct_coeffs[:50, :50]))
        features['dct_mid_energy'] = np.sum(np.abs(dct_coeffs[50:100, 50:100]))
        features['dct_high_energy'] = np.sum(np.abs(dct_coeffs[100:, 100:]))
        
        # Spectral entropy
        power_spectrum = magnitude_spectrum ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        features['spectral_entropy'] = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
        
        return features
    
    # Helper functions
    def split_into_blocks(self, img, block_size):
        """Split image into blocks"""
        blocks = []
        h, w = img.shape[:2]
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = img[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    blocks.append(block)
        return blocks
    
    def calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def calculate_entropy(self, img):
        """Calculate image entropy"""
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def calculate_lbp(self, img):
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(img)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                center = img[i, j]
                code = 0
                code |= (img[i-1, j-1] >= center) << 7
                code |= (img[i-1, j] >= center) << 6
                code |= (img[i-1, j+1] >= center) << 5
                code |= (img[i, j+1] >= center) << 4
                code |= (img[i+1, j+1] >= center) << 3
                code |= (img[i+1, j] >= center) << 2
                code |= (img[i+1, j-1] >= center) << 1
                code |= (img[i, j-1] >= center) << 0
                lbp[i, j] = code
        return lbp
    
    def apply_gabor_filters(self, img):
        """Apply Gabor filters for texture analysis"""
        features = {}
        
        # Gabor filter parameters
        ksize = 31
        sigma = 4.0
        lambd = 10.0
        gamma = 0.5
        
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            gabor_responses.append(filtered)
            
        # Extract statistics from Gabor responses
        for i, response in enumerate(gabor_responses):
            features[f'gabor_{i}_mean'] = np.mean(response)
            features[f'gabor_{i}_std'] = np.std(response)
            
        return features
    
    def calculate_glcm_features(self, img):
        """Calculate Gray Level Co-occurrence Matrix features"""
        features = {}
        
        # Quantize image to reduce computation
        img_quant = (img // 16).astype(np.uint8)
        
        # Calculate GLCM for horizontal direction
        glcm = np.zeros((16, 16))
        for i in range(img_quant.shape[0]):
            for j in range(img_quant.shape[1]-1):
                glcm[img_quant[i, j], img_quant[i, j+1]] += 1
                
        # Normalize
        glcm = glcm / glcm.sum()
        
        # Extract GLCM features
        features['glcm_contrast'] = np.sum((np.arange(16)[:, None] - np.arange(16)) ** 2 * glcm)
        features['glcm_homogeneity'] = np.sum(glcm / (1 + np.abs(np.arange(16)[:, None] - np.arange(16))))
        features['glcm_energy'] = np.sum(glcm ** 2)
        features['glcm_correlation'] = self.calculate_glcm_correlation(glcm)
        
        return features
    
    def calculate_glcm_correlation(self, glcm):
        """Calculate correlation from GLCM"""
        i_indices = np.arange(glcm.shape[0])[:, None]
        j_indices = np.arange(glcm.shape[1])
        
        mu_i = np.sum(i_indices * glcm)
        mu_j = np.sum(j_indices * glcm)
        
        sigma_i = np.sqrt(np.sum((i_indices - mu_i) ** 2 * glcm))
        sigma_j = np.sqrt(np.sum((j_indices - mu_j) ** 2 * glcm))
        
        if sigma_i == 0 or sigma_j == 0:
            return 0
            
        correlation = np.sum((i_indices - mu_i) * (j_indices - mu_j) * glcm) / (sigma_i * sigma_j)
        return correlation
    
    def estimate_noise(self, img):
        """Estimate noise level in image"""
        # Use Laplacian for noise estimation
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sigma = np.sqrt(np.pi / 2) * np.mean(np.abs(laplacian))
        return sigma
    
    def get_feature_vector(self, image_path):
        """
        Get feature vector as numpy array for model training
        """
        features = self.extract_all_features(image_path)
        
        # Convert to numpy array with consistent ordering
        feature_vector = []
        feature_names = []
        
        for key, value in sorted(features.items()):
            if not np.isnan(value) and not np.isinf(value):
                feature_vector.append(value)
                feature_names.append(key)
            else:
                feature_vector.append(0)
                feature_names.append(key)
        
        self.feature_names = feature_names
        return np.array(feature_vector)
    
    def get_feature_maps(self, image_path):
        """
        Generate feature visualization maps for display
        """
        img_bgr = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Generate PRNU noise map
        denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)
        prnu_map = img_gray.astype(float) - denoised.astype(float)
        prnu_map = np.abs(prnu_map)
        prnu_map = (prnu_map / prnu_map.max() * 255).astype(np.uint8)
        prnu_map = cv2.applyColorMap(prnu_map, cv2.COLORMAP_JET)
        
        # Generate ELA map
        img_pil = Image.open(image_path).convert('RGB')
        img_compressed = img_pil.copy()
        img_compressed.save('temp_ela.jpg', quality=90)
        compressed = Image.open('temp_ela.jpg')
        ela_img = ImageChops.difference(img_pil, compressed)
        
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff + 1e-10)
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        ela_map = np.array(ela_img)
        
        # Clean up
        import os
        if os.path.exists('temp_ela.jpg'):
            os.remove('temp_ela.jpg')
        
        return {
            'prnu': prnu_map,
            'ela': ela_map
        }


if __name__ == "__main__":
    # Test the feature extractor
    extractor = PixelWiseFeatureExtractor()
    print("Pixel-wise Feature Extractor initialized successfully!")
    print(f"Extracting features from images will provide {len(extractor.get_feature_vector('test.jpg'))} features")
