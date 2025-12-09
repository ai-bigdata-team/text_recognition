import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [images, setImages] = useState([])
  const [activeImageIndex, setActiveImageIndex] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const fileInputRef = useRef(null)

  const handleImageSelect = (files) => {
    const newImages = []
    Array.from(files).forEach((file, index) => {
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onloadend = () => {
          newImages.push({
            id: Date.now() + index,
            file: file,
            preview: reader.result,
            text: '',
            isProcessed: false
          })
          if (newImages.length === files.length) {
            setImages(prev => [...prev, ...newImages])
          }
        }
        reader.readAsDataURL(file)
      }
    })
  }

  const handleFileInput = (e) => {
    const files = e.target.files
    if (files && files.length > 0) handleImageSelect(files)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files && files.length > 0) handleImageSelect(files)
  }

  const handleRecognize = async () => {
    if (images.length === 0) return
    
    setIsProcessing(true)
    setProcessingProgress(0)
    
    // Simulate API call với progress - thay thế bằng API thực tế
    for (let i = 0; i < images.length; i++) {
      if (!images[i].isProcessed) {
        await new Promise(resolve => {
          setTimeout(() => {
            const mockText = `📄 Văn bản từ ảnh ${i + 1}\n\n✨ Đây là nội dung được nhận diện từ hình ảnh số ${i + 1}.\n\n🎯 Hệ thống đã phân tích và trích xuất văn bản với độ chính xác cao.\n\n💡 Bạn có thể thay thế đoạn code này bằng API thực tế để nhận diện chữ viết từ ảnh của bạn.\n\n🚀 Kết quả hiển thị với animation mượt mà và giao diện hiện đại!`
            
            setImages(prev => prev.map((img, idx) => 
              idx === i ? { ...img, text: mockText, isProcessed: true } : img
            ))
            setProcessingProgress(((i + 1) / images.length) * 100)
            resolve()
          }, 1500)
        })
      }
    }
    
    setIsProcessing(false)
    setProcessingProgress(0)
  }

  const handleReset = () => {
    setImages([])
    setActiveImageIndex(0)
    setIsProcessing(false)
    setProcessingProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleRemoveImage = (id) => {
    setImages(prev => prev.filter(img => img.id !== id))
    if (activeImageIndex >= images.length - 1) {
      setActiveImageIndex(Math.max(0, images.length - 2))
    }
  }

  const activeImage = images[activeImageIndex]
  const allText = images.filter(img => img.text).map(img => img.text).join('\n\n---\n\n')

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-black relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="floating-orb bg-purple-600/20 w-96 h-96 rounded-full blur-3xl absolute top-20 -left-20 animate-float"></div>
        <div className="floating-orb bg-pink-600/20 w-96 h-96 rounded-full blur-3xl absolute bottom-20 -right-20 animate-float-delayed"></div>
        <div className="floating-orb bg-blue-600/20 w-96 h-96 rounded-full blur-3xl absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse-slow"></div>
      </div>

      <div className="relative z-10 py-12 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12 animate-fade-in">
            <div className="inline-block mb-6 pb-2 animate-glow">
              <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent leading-tight animate-gradient">
                ✨ AI Text Recognition
              </h1>
            </div>
            <p className="text-gray-300 text-lg animate-fade-in-delay">
              Upload nhiều ảnh cùng lúc và xem phép màu AI xảy ra 🚀
            </p>
            <div className="flex gap-4 justify-center mt-6 text-sm text-gray-400 animate-fade-in-delay-2">
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                Hỗ trợ nhiều ảnh
              </span>
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></span>
                Drag & Drop
              </span>
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                Real-time Processing
              </span>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Upload Section - CỘT TRÁI */}
            <div className="space-y-6">
              <div
                className={`border-4 border-dashed rounded-2xl p-8 transition-all duration-300 backdrop-blur-xl ${
                  isDragging
                    ? 'border-purple-500 bg-purple-500/20 scale-105 shadow-2xl shadow-purple-500/50'
                    : 'border-purple-500/30 bg-gray-800/50 hover:border-purple-400/50 hover:bg-gray-800/70'
                } ${
                  images.length > 0 ? 'h-auto' : 'h-96'
                } flex flex-col items-center justify-center cursor-pointer hover:shadow-xl hover:shadow-purple-500/30 animate-border-glow`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                {images.length === 0 ? (
                  <>
                    <div className="text-7xl mb-6 animate-float">📸</div>
                    <h3 className="text-2xl font-semibold text-white mb-3 animate-pulse-slow">
                      Kéo thả nhiều ảnh vào đây
                    </h3>
                    <p className="text-purple-300 mb-6 text-lg">hoặc</p>
                    <button className="px-8 py-4 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 text-white rounded-xl font-medium hover:scale-110 transition-all shadow-lg shadow-purple-500/50 hover:shadow-purple-500/70 animate-gradient-x relative overflow-hidden group">
                      <span className="relative z-10 flex items-center gap-2">
                        <span>🚀</span>
                        Chọn ảnh từ thiết bị
                      </span>
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    </button>
                    <p className="text-gray-400 text-sm mt-4">Hỗ trợ: JPG, PNG, GIF • Nhiều ảnh cùng lúc</p>
                  </>
                ) : (
                  <div className="w-full animate-scale-in space-y-4" onClick={(e) => e.stopPropagation()}>
                    {/* Main Image */}
                    {activeImage && (
                      <div className="relative group">
                        <img
                          src={activeImage.preview}
                          alt="Preview"
                          className="w-full h-auto rounded-xl shadow-2xl shadow-purple-500/30 border-2 border-purple-500/50"
                        />
                        <div className="absolute top-3 right-3 flex gap-2">
                          <button
                            onClick={() => handleRemoveImage(activeImage.id)}
                            className="p-2 bg-red-500/80 hover:bg-red-500 text-white rounded-lg backdrop-blur-sm transition-all"
                          >
                            🗑️
                          </button>
                        </div>
                        {activeImage.isProcessed && (
                          <div className="absolute top-3 left-3">
                            <span className="px-3 py-1 bg-green-500/90 text-white rounded-lg text-sm font-medium backdrop-blur-sm animate-fade-in">
                              ✓ Đã xử lý
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Thumbnails */}
                    {images.length > 1 && (
                      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-gray-700">
                        {images.map((img, idx) => (
                          <div
                            key={img.id}
                            onClick={() => setActiveImageIndex(idx)}
                            className={`relative flex-shrink-0 cursor-pointer transition-all duration-300 ${
                              idx === activeImageIndex
                                ? 'ring-4 ring-purple-500 scale-110'
                                : 'hover:scale-105 opacity-60 hover:opacity-100'
                            }`}
                          >
                            <img
                              src={img.preview}
                              alt={`Thumbnail ${idx + 1}`}
                              className="w-20 h-20 object-cover rounded-lg"
                            />
                            {img.isProcessed && (
                              <div className="absolute -top-1 -right-1 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center text-xs animate-bounce">
                                ✓
                              </div>
                            )}
                          </div>
                        ))}
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="flex-shrink-0 w-20 h-20 border-2 border-dashed border-purple-500/50 rounded-lg flex items-center justify-center text-3xl hover:bg-purple-500/20 transition-all hover:scale-105"
                        >
                          ➕
                        </button>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                      <button
                        onClick={handleRecognize}
                        disabled={isProcessing}
                        className="flex-1 px-6 py-4 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 text-white rounded-xl font-medium hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-green-500/30 hover:shadow-green-500/50 relative overflow-hidden group"
                      >
                        <span className="relative z-10 flex items-center justify-center gap-2">
                          {isProcessing ? (
                            <>
                              <span className="animate-spin">⚙️</span>
                              Đang xử lý {Math.round(processingProgress)}%
                            </>
                          ) : (
                            <>
                              <span>🚀</span>
                              Nhận diện {images.length} ảnh
                            </>
                          )}
                        </span>
                        {isProcessing && (
                          <div 
                            className="absolute bottom-0 left-0 h-1 bg-white/50 transition-all duration-300"
                            style={{ width: `${processingProgress}%` }}
                          />
                        )}
                      </button>
                      <button
                        onClick={handleReset}
                        className="px-6 py-4 bg-gray-700/50 hover:bg-gray-700 text-white rounded-xl font-medium transition-all backdrop-blur-sm border border-gray-600 hover:scale-105"
                      >
                        🔄 Reset
                      </button>
                    </div>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileInput}
                  className="hidden"
                />
              </div>

              {/* Processing Animation */}
              {isProcessing && (
                <div className="bg-gradient-to-br from-purple-900/50 to-blue-900/50 rounded-2xl p-8 backdrop-blur-xl border border-purple-500/30 animate-scale-in">
                  <div className="flex items-center justify-center space-x-3 mb-4">
                    <div className="w-4 h-4 bg-purple-500 rounded-full animate-bounce shadow-lg shadow-purple-500/50" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-4 h-4 bg-pink-500 rounded-full animate-bounce shadow-lg shadow-pink-500/50" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-4 h-4 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <p className="text-center text-white font-medium mb-2">🤖 AI đang phân tích hình ảnh...</p>
                  <div className="w-full bg-gray-700/50 rounded-full h-3 overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 transition-all duration-300 animate-gradient-x"
                      style={{ width: `${processingProgress}%` }}
                    />
                  </div>
                  <p className="text-center text-gray-400 text-sm mt-2">{Math.round(processingProgress)}% hoàn thành</p>
                </div>
              )}

              {/* Stats Cards */}
              {images.length > 0 && (
                <div className="grid grid-cols-3 gap-4 animate-fade-in">
                  <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-4 backdrop-blur-xl border border-purple-500/30 hover:scale-105 transition-all">
                    <div className="text-3xl font-bold text-purple-300">{images.length}</div>
                    <div className="text-sm text-gray-400">Ảnh đã tải</div>
                  </div>
                  <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-4 backdrop-blur-xl border border-green-500/30 hover:scale-105 transition-all">
                    <div className="text-3xl font-bold text-green-300">{images.filter(img => img.isProcessed).length}</div>
                    <div className="text-sm text-gray-400">Đã xử lý</div>
                  </div>
                  <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-4 backdrop-blur-xl border border-blue-500/30 hover:scale-105 transition-all">
                    <div className="text-3xl font-bold text-blue-300">{images.filter(img => !img.isProcessed).length}</div>
                    <div className="text-sm text-gray-400">Chờ xử lý</div>
                  </div>
                </div>
              )}

              {/* Tips - PHẦN NÀY Ở BÊN TRÁI */}
              <div className="bg-gradient-to-br from-purple-900/50 via-pink-900/30 to-blue-900/50 rounded-2xl p-6 backdrop-blur-xl border border-purple-500/30 hover:shadow-xl hover:shadow-purple-500/20 transition-all animate-fade-in">
                <h3 className="font-semibold text-white mb-4 flex items-center gap-2 text-lg">
                  <span className="animate-pulse-slow">💡</span>
                  Gợi ý để có kết quả tốt nhất
                </h3>
                <ul className="space-y-3 text-sm text-gray-300">
                  <li className="flex items-start gap-3 group">
                    <span className="text-green-400 mt-0.5 group-hover:scale-125 transition-transform">✓</span>
                    <span className="group-hover:text-white transition-colors">Sử dụng ảnh có độ phân giải cao và rõ nét</span>
                  </li>
                  <li className="flex items-start gap-3 group">
                    <span className="text-green-400 mt-0.5 group-hover:scale-125 transition-transform">✓</span>
                    <span className="group-hover:text-white transition-colors">Đảm bảo chữ viết rõ ràng và có đủ ánh sáng</span>
                  </li>
                  <li className="flex items-start gap-3 group">
                    <span className="text-green-400 mt-0.5 group-hover:scale-125 transition-transform">✓</span>
                    <span className="group-hover:text-white transition-colors">Tránh ảnh bị mờ hoặc nghiêng quá nhiều</span>
                  </li>
                  <li className="flex items-start gap-3 group">
                    <span className="text-green-400 mt-0.5 group-hover:scale-125 transition-transform">✓</span>
                    <span className="group-hover:text-white transition-colors">Upload nhiều ảnh cùng lúc để xử lý nhanh hơn</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Result Section - CỘT PHẢI */}
            <div className="space-y-6">
              <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl p-8 border border-purple-500/30 min-h-96 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/20 hover:border-purple-500/50">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-3xl font-bold text-white flex items-center gap-3">
                    <span className="animate-pulse-slow">📝</span>
                    Kết quả nhận diện
                  </h2>
                  <div className="flex gap-2">
                    {activeImage && activeImage.text && (
                      <button
                        onClick={() => navigator.clipboard.writeText(activeImage.text)}
                        className="px-4 py-2 bg-blue-600/80 hover:bg-blue-600 text-white rounded-lg transition-all text-sm font-medium backdrop-blur-sm hover:scale-105 shadow-lg shadow-blue-500/30"
                      >
                        📋 Copy ảnh này
                      </button>
                    )}
                    {allText && (
                      <button
                        onClick={() => navigator.clipboard.writeText(allText)}
                        className="px-4 py-2 bg-purple-600/80 hover:bg-purple-600 text-white rounded-lg transition-all text-sm font-medium backdrop-blur-sm hover:scale-105 shadow-lg shadow-purple-500/30"
                      >
                        📋 Copy tất cả
                      </button>
                    )}
                  </div>
                </div>
                
                {!activeImage?.text && !isProcessing ? (
                  <div className="flex flex-col items-center justify-center h-64 text-gray-400 animate-fade-in">
                    <div className="text-7xl mb-6 animate-float">📄</div>
                    <p className="text-center text-lg">Văn bản nhận diện sẽ xuất hiện ở đây</p>
                    <p className="text-center text-sm text-gray-500 mt-2">Upload ảnh và nhấn "Nhận diện" để bắt đầu ✨</p>
                  </div>
                ) : activeImage?.text ? (
                  <div className="animate-slide-up space-y-4">
                    {/* Tab selector for multiple results */}
                    {images.filter(img => img.text).length > 1 && (
                      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-gray-700">
                        {images.map((img, idx) => (
                          img.text && (
                            <button
                              key={img.id}
                              onClick={() => setActiveImageIndex(idx)}
                              className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                                idx === activeImageIndex
                                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50'
                                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-700'
                              }`}
                            >
                              Ảnh {idx + 1}
                            </button>
                          )
                        ))}
                      </div>
                    )}

                    <div className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 rounded-xl p-6 border-2 border-purple-500/30 backdrop-blur-sm max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-gray-700">
                      <pre className="whitespace-pre-wrap text-gray-200 text-lg leading-relaxed font-sans">
                        {activeImage.text}
                      </pre>
                    </div>
                    
                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-4 text-center backdrop-blur-sm border border-purple-500/30 hover:scale-105 transition-all">
                        <div className="text-3xl font-bold text-purple-300 animate-fade-in">
                          {activeImage.text.length}
                        </div>
                        <div className="text-sm text-gray-400">Ký tự</div>
                      </div>
                      <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-4 text-center backdrop-blur-sm border border-blue-500/30 hover:scale-105 transition-all">
                        <div className="text-3xl font-bold text-blue-300 animate-fade-in">
                          {activeImage.text.split(/\s+/).filter(w => w.length > 0).length}
                        </div>
                        <div className="text-sm text-gray-400">Từ</div>
                      </div>
                      <div className="bg-gradient-to-br from-pink-900/50 to-pink-800/30 rounded-xl p-4 text-center backdrop-blur-sm border border-pink-500/30 hover:scale-105 transition-all">
                        <div className="text-3xl font-bold text-pink-300 animate-fade-in">
                          {activeImage.text.split('\n').filter(l => l.trim().length > 0).length}
                        </div>
                        <div className="text-sm text-gray-400">Dòng</div>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>

              {/* Additional Features - PHẦN NÀY Ở BÊN PHẢI */}
              <div className="grid grid-cols-2 gap-4 animate-fade-in-delay">
                <div className="bg-gradient-to-br from-green-900/50 to-emerald-900/30 rounded-xl p-4 backdrop-blur-xl border border-green-500/30 hover:scale-105 transition-all">
                  <div className="text-3xl mb-2">⚡</div>
                  <div className="text-white font-medium text-sm">Xử lý siêu nhanh</div>
                  <div className="text-gray-400 text-xs mt-1">AI tối ưu hóa</div>
                </div>
                <div className="bg-gradient-to-br from-blue-900/50 to-cyan-900/30 rounded-xl p-4 backdrop-blur-xl border border-blue-500/30 hover:scale-105 transition-all">
                  <div className="text-3xl mb-2">🎯</div>
                  <div className="text-white font-medium text-sm">Độ chính xác cao</div>
                  <div className="text-gray-400 text-xs mt-1">Deep Learning</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
