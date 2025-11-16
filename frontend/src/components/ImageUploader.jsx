import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const ImageUploader = ({ onUpload }) => {
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Trigger upload
      onUpload(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff']
    },
    multiple: false,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false),
    onDropAccepted: () => setIsDragging(false),
  });

  const clearPreview = () => {
    setPreview(null);
  };

  return (
    <div className="w-full">
      <motion.div
        {...getRootProps()}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        animate={{
          borderColor: isDragging ? 'rgb(147, 51, 234)' : 'rgba(255, 255, 255, 0.2)',
          backgroundColor: isDragging ? 'rgba(147, 51, 234, 0.1)' : 'rgba(255, 255, 255, 0.05)',
        }}
        className={`
          relative cursor-pointer rounded-2xl p-12
          border-2 border-dashed transition-all duration-300
          backdrop-blur-md bg-white/5 hover:bg-white/10
          ${isDragActive ? 'border-purple-500 bg-purple-500/10' : 'border-white/20'}
        `}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center space-y-4"
            >
              <motion.div
                animate={{
                  y: isDragging ? -10 : 0,
                  scale: isDragging ? 1.1 : 1,
                }}
                className="p-4 rounded-full bg-gradient-to-br from-blue-500 to-purple-600"
              >
                {isDragging ? (
                  <Image className="w-12 h-12 text-white" />
                ) : (
                  <Upload className="w-12 h-12 text-white" />
                )}
              </motion.div>
              
              <div className="text-center">
                <p className="text-lg font-medium text-white mb-2">
                  {isDragActive ? 'Drop your image here' : 'Upload an image to analyze'}
                </p>
                <p className="text-sm text-gray-400">
                  Drag & drop or click to browse
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Supports PNG, JPEG, GIF, BMP, WebP, TIFF
                </p>
              </div>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium shadow-lg hover:shadow-xl transition-shadow"
                onClick={(e) => e.stopPropagation()}
              >
                Choose Image
              </motion.button>
            </motion.div>
          ) : (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="relative"
            >
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg shadow-2xl"
              />
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={(e) => {
                  e.stopPropagation();
                  clearPreview();
                }}
                className="absolute top-2 right-2 p-2 bg-red-500/80 rounded-full text-white hover:bg-red-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Upload hints */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-effect p-4 rounded-lg">
          <h4 className="text-sm font-semibold text-blue-400 mb-1">Real Images</h4>
          <p className="text-xs text-gray-400">
            Photos taken with cameras or smartphones
          </p>
        </div>
        <div className="glass-effect p-4 rounded-lg">
          <h4 className="text-sm font-semibold text-purple-400 mb-1">AI-Generated</h4>
          <p className="text-xs text-gray-400">
            Images created by AI models like DALL-E, Midjourney
          </p>
        </div>
        <div className="glass-effect p-4 rounded-lg">
          <h4 className="text-sm font-semibold text-green-400 mb-1">Max Size</h4>
          <p className="text-xs text-gray-400">
            16MB maximum file size supported
          </p>
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;
