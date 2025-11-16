import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Shield, AlertTriangle, CheckCircle, Info, Zap, Eye, Brain } from 'lucide-react';
import ImageUploader from './components/ImageUploader';
import AnalysisResult from './components/AnalysisResult';
import FeatureVisualization from './components/FeatureVisualization';
import Header from './components/Header';
import LoadingAnimation from './components/LoadingAnimation';
import axios from 'axios';

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage(reader.result);
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    try {
      // Use the full URL to the Flask backend
      const response = await axios.post('http://localhost:5000/api/upload_and_analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAnalysisResult(response.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.error || 'Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setUploadedImage(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-6xl mx-auto"
        >
          {/* Hero Section */}
          <div className="text-center mb-12">
            <motion.h1
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-5xl font-bold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400"
            >
              AI-Generated Image Detection
            </motion.h1>
            <p className="text-gray-300 text-lg max-w-2xl mx-auto">
              Using advanced pixel-wise feature extraction and deep learning to detect AI-generated images
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="glass-effect p-6 rounded-xl text-white"
            >
              <Brain className="w-10 h-10 mb-3 text-blue-400" />
              <h3 className="font-semibold mb-2">Deep Learning Analysis</h3>
              <p className="text-sm text-gray-300">
                CNN-based model trained on thousands of images
              </p>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.05 }}
              className="glass-effect p-6 rounded-xl text-white"
            >
              <Eye className="w-10 h-10 mb-3 text-purple-400" />
              <h3 className="font-semibold mb-2">Pixel-wise Features</h3>
              <p className="text-sm text-gray-300">
                PRNU, ELA, and correlation analysis
              </p>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.05 }}
              className="glass-effect p-6 rounded-xl text-white"
            >
              <Zap className="w-10 h-10 mb-3 text-green-400" />
              <h3 className="font-semibold mb-2">Real-time Detection</h3>
              <p className="text-sm text-gray-300">
                Fast and accurate results in seconds
              </p>
            </motion.div>
          </div>

          {/* Main Content */}
          <AnimatePresence mode="wait">
            {!analysisResult && !isAnalyzing && (
              <motion.div
                key="uploader"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <ImageUploader onUpload={handleImageUpload} />
                
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200"
                  >
                    <AlertTriangle className="inline w-5 h-5 mr-2" />
                    {error}
                  </motion.div>
                )}
              </motion.div>
            )}

            {isAnalyzing && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <LoadingAnimation />
              </motion.div>
            )}

            {analysisResult && !isAnalyzing && (
              <motion.div
                key="result"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                {analysisResult && (
                  <AnalysisResult
                    result={analysisResult}
                    uploadedImage={uploadedImage}
                    onReset={resetAnalysis}
                  />
                )}

                {analysisResult?.visualizations && (
                  <FeatureVisualization visualizations={analysisResult.visualizations} />
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Info Section */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-16 glass-effect rounded-xl p-8 text-white"
          >
            <h2 className="text-2xl font-bold mb-4 flex items-center">
              <Info className="mr-2" />
              How It Works
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-2 text-blue-400">Detection Techniques</h3>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• Photo Response Non-Uniformity (PRNU) analysis</li>
                  <li>• Error Level Analysis (ELA) for compression artifacts</li>
                  <li>• Inter-pixel correlation patterns</li>
                  <li>• Texture and noise feature extraction</li>
                  <li>• Frequency domain analysis</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-purple-400">Key Features</h3>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• Deep CNN architecture for image analysis</li>
                  <li>• Hybrid model combining pixel features and CNN</li>
                  <li>• Real-time processing and visualization</li>
                  <li>• Confidence scores and risk assessment</li>
                  <li>• Feature map visualization (PRNU, ELA)</li>
                </ul>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </main>
    </div>
  );
}

export default App;
