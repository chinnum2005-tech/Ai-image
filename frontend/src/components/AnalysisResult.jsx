import React from 'react';
import { motion } from 'framer-motion';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  Info,
  Shield,
  RefreshCw,
  Download,
  TrendingUp,
  Camera,
  Cpu
} from 'lucide-react';

// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error in AnalysisResult:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-200">
          <AlertTriangle className="inline w-5 h-5 mr-2" />
          Something went wrong while displaying the analysis result.
        </div>
      );
    }
    return this.props.children;
  }
}

// Main component
const AnalysisResult = ({ result = {}, uploadedImage, onReset }) => {
  // Ensure result is an object
  if (!result || typeof result !== 'object') {
    return (
      <div className="p-4 bg-yellow-900/20 border border-yellow-500/50 rounded-lg text-yellow-200">
        <AlertTriangle className="inline w-5 h-5 mr-2" />
        No analysis data available
      </div>
    );
  }

  // Handle both test server and real server response formats
  let processedResult = result;
  
  // If this is a test server response (has 'status' field), extract the data
  if (result.status === 'success' && result.prediction) {
    processedResult = {
      ...result,
      prediction: {
        ...result.prediction,
        result_text: result.prediction.class || (result.prediction.is_ai_generated ? 'AI-Generated Image' : 'Real Photograph'),
        risk_level: result.analysis?.risk_level || 'UNKNOWN',
        raw_score: result.prediction.confidence || 0
      },
      analysis: {
        ...result.analysis,
        summary: result.analysis?.details || 'No analysis details available',
        key_indicators: result.features ? Object.entries(result.features.key_indicators || {}).map(([key, value]) => 
          `${key.replace('_', ' ')}: ${value}`) : []
      }
    };
  }

  // Safely extract values with defaults
  const prediction = {
    confidence: 0,
    is_ai_generated: false,
    result_text: 'Unknown',
    risk_level: 'UNKNOWN',
    raw_score: 0,
    message: 'No prediction available',
    ...(processedResult.prediction || {})
  };

  const analysis = {
    risk_level: 'UNKNOWN',
    summary: '',
    key_indicators: [],
    recommendations: [],
    ...(processedResult.analysis || {})
  };

  const image_info = {
    size_kb: 0,
    width: 0,
    height: 0,
    format: 'UNKNOWN',
    filename: 'unknown',
    ...(processedResult.image_info || processedResult.features || {})
  };
  
  const confidencePercent = Math.round((prediction.confidence || 0) * 100);
  
  const getRiskColor = (level) => {
    const colors = {
      CRITICAL: 'text-red-500',
      HIGH: 'text-orange-500',
      MEDIUM: 'text-yellow-500',
      LOW: 'text-green-500'
    };
    return colors[level] || 'text-gray-500';
  };

  const getRiskIcon = (level) => {
    const icons = {
      CRITICAL: XCircle,
      HIGH: AlertTriangle,
      MEDIUM: Info,
      LOW: CheckCircle
    };
    return icons[level] || Info;
  };

  const RiskIcon = getRiskIcon(prediction.risk_level);

  return (
    <div className="space-y-6">
      {/* Main Result Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-effect rounded-2xl p-8"
      >
        <div className="grid md:grid-cols-2 gap-8">
          {/* Image Preview */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <Camera className="w-5 h-5 mr-2" />
              Analyzed Image
            </h3>
            <div className="relative rounded-lg overflow-hidden shadow-2xl">
              {uploadedImage ? (
                <img
                  src={uploadedImage}
                  alt="Analyzed"
                  className="w-full h-auto"
                />
              ) : (
                <div className="w-full h-48 bg-gray-700 flex items-center justify-center">
                  <Camera className="w-12 h-12 text-gray-500" />
                </div>
              )}
              <div className="absolute top-2 right-2 px-3 py-1 bg-black/50 backdrop-blur-sm rounded-full">
                <span className="text-xs text-white">
                  {Math.round(image_info.size_kb || 0)}KB
                </span>
              </div>
            </div>
            <div className="flex justify-between text-sm text-gray-400">
              <span>{(image_info.width || 0)} × {(image_info.height || 0)}px</span>
              <span>{image_info.filename || 'Unknown'}</span>
            </div>
          </div>

          {/* Analysis Results */}
          <div className="space-y-6">
            <div className="text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200 }}
                className="w-40 h-40 mx-auto mb-4"
              >
                <CircularProgressbar
                  value={confidencePercent}
                  text={`${confidencePercent}%`}
                  styles={buildStyles({
                    textSize: '20px',
                    pathColor: prediction.is_ai_generated
                      ? `rgba(239, 68, 68, ${prediction.confidence})`
                      : `rgba(34, 197, 94, ${prediction.confidence})`,
                    textColor: '#fff',
                    trailColor: 'rgba(255, 255, 255, 0.1)',
                  })}
                />
              </motion.div>
              
              <motion.h2
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className={`text-2xl font-bold mb-2 ${
                  prediction.is_ai_generated ? 'text-red-400' : 'text-green-400'
                }`}
              >
                {prediction.result_text}
              </motion.h2>
              
              <div className="flex items-center justify-center space-x-2">
                <RiskIcon className={`w-5 h-5 ${getRiskColor(prediction.risk_level)}`} />
                <span className={`font-semibold ${getRiskColor(prediction.risk_level)}`}>
                  {prediction.risk_level} RISK
                </span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="glass-effect rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Confidence Score</span>
                  <span className="text-sm font-semibold text-white">
                    {((prediction.confidence || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              
              <div className="glass-effect rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Raw Prediction</span>
                  <span className="text-sm font-semibold text-white">
                    {(prediction.raw_score || 0).toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Analysis Details */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-effect rounded-2xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Shield className="w-5 h-5 mr-2" />
          Detailed Analysis
        </h3>
        
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-semibold text-blue-400 mb-2">Summary</h4>
            <p className="text-sm text-gray-300">{analysis.summary || 'No analysis summary available'}</p>
          </div>
          
          {analysis.key_indicators && analysis.key_indicators.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-purple-400 mb-2">Key Indicators</h4>
              <ul className="space-y-1">
                {analysis.key_indicators.map((indicator, idx) => (
                  <li key={idx} className="text-sm text-gray-300 flex items-start">
                    <TrendingUp className="w-4 h-4 mr-2 mt-0.5 text-purple-400 flex-shrink-0" />
                    {indicator}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {analysis.recommendations && analysis.recommendations.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-green-400 mb-2">Recommendations</h4>
              <ul className="space-y-1">
                {analysis.recommendations.map((rec, idx) => (
                  <li key={idx} className="text-sm text-gray-300 flex items-start">
                    <Info className="w-4 h-4 mr-2 mt-0.5 text-green-400 flex-shrink-0" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </motion.div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="flex flex-wrap gap-4 justify-center"
      >
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onReset}
          className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium shadow-lg hover:shadow-xl transition-shadow flex items-center"
        >
          <RefreshCw className="w-5 h-5 mr-2" />
          Analyze Another Image
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-6 py-3 glass-effect text-white rounded-lg font-medium hover:bg-white/20 transition-colors flex items-center"
          onClick={() => {
            const dataStr = JSON.stringify(processedResult, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
            const link = document.createElement('a');
            link.setAttribute('href', dataUri);
            link.setAttribute('download', `analysis_${processedResult.id || 'result'}.json`);
            link.click();
          }}
        >
          <Download className="w-5 h-5 mr-2" />
          Download Full Report
        </motion.button>
      </motion.div>
    </div>
  );
};

// Wrap the component with ErrorBoundary
export default function WrappedAnalysisResult(props) {
  return (
    <ErrorBoundary>
      <AnalysisResult {...props} />
    </ErrorBoundary>
  );
}
