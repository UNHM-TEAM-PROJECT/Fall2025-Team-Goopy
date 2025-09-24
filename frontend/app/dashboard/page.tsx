"use client";

import React, { useState, useEffect } from "react";

export default function TestResultsPage() {
  const [testData, setTestData] = useState<any>(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [testRunMessage, setTestRunMessage] = useState<string | null>(null);

  useEffect(() => {
    // Load test results data
    const loadData = async () => {
      try {
        const response = await fetch('http://localhost:8003/test-results');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setTestData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to load test data:', err);
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
        setError(`Failed to connect to backend: ${errorMessage}`);
        setTestData(null);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handleRunNewTest = async () => {
    setIsRunningTest(true);
    setTestRunMessage("Initiating new test run...");
    
    try {
      const response = await fetch('http://localhost:8003/run-tests', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setTestRunMessage("Test completed successfully! Refreshing data...");
      
      // Refresh the data after a successful test run
      setTimeout(async () => {
        try {
          const response = await fetch('http://localhost:8003/test-results');
          if (response.ok) {
            const data = await response.json();
            setTestData(data);
            setTestRunMessage("Data refreshed successfully!");
            setTimeout(() => setTestRunMessage(null), 3000);
          }
        } catch (err) {
          console.error('Failed to refresh data:', err);
          setTestRunMessage("Test completed, but failed to refresh data. Please refresh manually.");
          setTimeout(() => setTestRunMessage(null), 5000);
        }
      }, 2000);
      
    } catch (err) {
      console.error('Failed to run test:', err);
      setTestRunMessage("Failed to run test. Please check backend connection.");
      setTimeout(() => setTestRunMessage(null), 5000);
    } finally {
      setIsRunningTest(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[var(--unh-white)] flex items-center justify-center">
        <div className="text-xl">Loading test results...</div>
      </div>
    );
  }

  if (error || !testData) {
    return (
      <main className="min-h-screen bg-[var(--unh-white)]">
        <header className="bg-[var(--unh-blue)] px-8 py-4 text-center shadow-md" style={{ color: '#fff' }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <img 
                src="/unh.svg" 
                alt="UNH Logo" 
                className="my-6 mr-4" 
                style={{ maxWidth: '125px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} 
              />
              <span className="text-3xl font-bold whitespace-nowrap" style={{ fontFamily: 'Glypha, Arial, sans-serif' }}>
                Test Dashboard
              </span>
            </div>
          </div>
        </header>

        <div className="container mx-auto px-8 py-8">
          <div className="text-center">
            <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded max-w-lg mx-auto">
              <strong>Error:</strong> {error || 'Unable to load test results'}
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[var(--unh-white)] overflow-auto">
      <header className="bg-[var(--unh-blue)] px-8 py-4 text-center shadow-md sticky top-0 z-10" style={{ color: '#fff' }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <img 
              src="/unh.svg" 
              alt="UNH Logo" 
              className="my-6 mr-4" 
              style={{ maxWidth: '125px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} 
            />
            <div className="text-left">
              <span className="text-3xl font-bold whitespace-nowrap block" style={{ fontFamily: 'Glypha, Arial, sans-serif' }}>
                Test Dashboard
              </span>
              <p className="text-blue-100 text-sm mt-1 whitespace-nowrap">View and compare automated testing results across multiple report runs</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            {isRunningTest ? (
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent mr-3"></div>
                <span className="text-sm">Running Test...</span>
              </div>
            ) : (
              <button
                onClick={handleRunNewTest}
                className="bg-white text-[var(--unh-blue)] px-6 py-2 rounded-lg hover:bg-blue-50 transition-colors font-medium"
              >
                Start Test Run
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto px-8 py-8 pb-16">

        {/* Reports Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          {/* Generate multiple report cards using the same data */}
          {[1, 2, 3, 4, 5, 6].map((reportIndex) => (
            <div key={reportIndex} className="bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
              {/* Report Card Header */}
              <div className="bg-[var(--unh-blue)] text-white p-6">
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-xl font-bold mb-1">Report #{reportIndex}</h2>
                    <p className="text-blue-100 text-sm">
                      {new Date(new Date(testData.lastRun).getTime() - (reportIndex - 1) * 24 * 60 * 60 * 1000).toLocaleDateString()} at{' '}
                      {new Date(new Date(testData.lastRun).getTime() - (reportIndex - 1) * 24 * 60 * 60 * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">
                      {(testData.summary.bertscore_f1 * 100).toFixed(1)}%
                    </div>
                    <div className="text-blue-100 text-xs">Overall Score</div>
                  </div>
                </div>
              </div>

              {/* Report Summary Metrics */}
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h5 className="text-sm font-semibold text-gray-700 mb-2">SBERT Cosine</h5>
                    <p className="text-2xl font-bold text-green-600">
                      {(testData.summary.sbert_cosine * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Sentence similarity</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h5 className="text-sm font-semibold text-gray-700 mb-2">Recall@1</h5>
                    <p className="text-2xl font-bold text-purple-600">
                      {(testData.summary["recall@1"] * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Top result accuracy</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h5 className="text-sm font-semibold text-gray-700 mb-2">NDCG@3</h5>
                    <p className="text-2xl font-bold text-orange-600">
                      {(testData.summary["ndcg@3"] * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Ranking quality</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h5 className="text-sm font-semibold text-gray-700 mb-2">Nugget F1</h5>
                    <p className="text-2xl font-bold text-red-600">
                      {(testData.summary.nugget_f1 * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Nugget precision & recall</p>
                  </div>
                </div>

                {/* Quick Stats */}
                <div className="border-t border-gray-200 pt-4">
                  <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span>Questions Tested:</span>
                    <span className="font-medium">{testData.summary.count}</span>
                  </div>
                  <div className="flex justify-between text-sm text-gray-600 mb-4">
                    <span>Categories:</span>
                    <span className="font-medium">
                      {testData.predictions_data ? Object.keys(testData.predictions_data.categories).length : 4}
                    </span>
                  </div>
                  
                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <button 
                      onClick={() => {
                        const reportData = { ...testData, selectedReport: reportIndex };
                        setSelectedReport(reportData);
                      }}
                      className="flex-1 bg-[var(--unh-blue)] text-white px-4 py-2 rounded-lg hover:bg-[var(--unh-accent-blue)] transition-colors text-sm font-medium"
                    >
                      View Details
                    </button>
                    <button className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium">
                      Compare
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Selected Report Detail Modal */}
        {selectedReport && (
          <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] overflow-hidden w-full">
              <div className="max-h-[90vh] overflow-y-auto">
                <div className="sticky top-0 bg-[var(--unh-blue)] border-b border-blue-600 p-6 z-20">
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 className="text-2xl font-bold text-white mb-2">
                        Report #{selectedReport.selectedReport} - Detailed View
                      </h2>
                      <p className="text-blue-100 text-sm mb-1">Last run: {new Date(selectedReport.lastRun).toLocaleString()}</p>
                      <p className="text-blue-100 text-sm">Total questions evaluated: {selectedReport.summary.count}</p>
                    </div>
                    <button 
                      onClick={() => setSelectedReport(null)}
                      className="text-blue-200 hover:text-white text-2xl font-bold"
                    >
                      ×
                  </button>
                </div>
              </div>
              
              <div className="p-6">
                {/* BERTscore Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">BERTscore F1</h3>
                    <p className="text-2xl font-bold text-blue-600">
                      {(selectedReport.summary.bertscore_f1 * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Semantic similarity</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">SBERT Cosine</h3>
                    <p className="text-2xl font-bold text-green-600">
                      {(selectedReport.summary.sbert_cosine * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Sentence similarity</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">Recall@1</h3>
                    <p className="text-2xl font-bold text-purple-600">
                      {(selectedReport.summary["recall@1"] * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Top result accuracy</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">NDCG@3</h3>
                    <p className="text-2xl font-bold text-orange-600">
                      {(selectedReport.summary["ndcg@3"] * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Ranking quality</p>
                  </div>
                </div>

                {/* Detailed Summary Metrics with Circular Progress */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8 relative z-10">
                  {/* Nugget Metrics */}
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
                    <h5 className="text-lg font-bold text-gray-800 mb-4">Nugget Metrics</h5>
                    <div className="flex justify-around items-center">
                      {/* Precision Circle */}
                      <div className="text-center">
                        <div className="relative inline-flex items-center justify-center w-16 h-16">
                          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="2"
                            />
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#059669"
                              strokeWidth="2"
                              strokeDasharray={`${selectedReport.summary.nugget_precision * 100}, 100`}
                            />
                          </svg>
                          <span className="absolute text-sm font-bold text-green-600">
                            {(selectedReport.summary.nugget_precision * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">Precision</div>
                      </div>
                      
                      {/* Recall Circle */}
                      <div className="text-center">
                        <div className="relative inline-flex items-center justify-center w-16 h-16">
                          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="2"
                            />
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#dc2626"
                              strokeWidth="2"
                              strokeDasharray={`${selectedReport.summary.nugget_recall * 100}, 100`}
                            />
                          </svg>
                          <span className="absolute text-sm font-bold text-red-600">
                            {(selectedReport.summary.nugget_recall * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">Recall</div>
                      </div>
                      
                      {/* F1 Score Circle */}
                      <div className="text-center">
                        <div className="relative inline-flex items-center justify-center w-16 h-16">
                          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="2"
                            />
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#2563eb"
                              strokeWidth="2"
                              strokeDasharray={`${selectedReport.summary.nugget_f1 * 100}, 100`}
                            />
                          </svg>
                          <span className="absolute text-sm font-bold text-blue-600">
                            {(selectedReport.summary.nugget_f1 * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">F1 Score</div>
                      </div>
                    </div>
                  </div>

                  {/* Ranking Metrics */}
                  <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
                    <h5 className="text-lg font-bold text-gray-800 mb-4">Ranking Performance</h5>
                    <div className="flex justify-around items-center">
                      {/* Recall@3 Circle */}
                      <div className="text-center">
                        <div className="relative inline-flex items-center justify-center w-16 h-16">
                          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="2"
                            />
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#059669"
                              strokeWidth="2"
                              strokeDasharray={`${selectedReport.summary["recall@3"] * 100}, 100`}
                            />
                          </svg>
                          <span className="absolute text-sm font-bold text-green-600">
                            {(selectedReport.summary["recall@3"] * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">Recall@3</div>
                      </div>
                      
                      {/* Recall@5 Circle */}
                      <div className="text-center">
                        <div className="relative inline-flex items-center justify-center w-16 h-16">
                          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="2"
                            />
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#059669"
                              strokeWidth="2"
                              strokeDasharray={`${selectedReport.summary["recall@5"] * 100}, 100`}
                            />
                          </svg>
                          <span className="absolute text-sm font-bold text-green-600">
                            {(selectedReport.summary["recall@5"] * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">Recall@5</div>
                      </div>
                      
                      {/* NDCG@5 Circle */}
                      <div className="text-center">
                        <div className="relative inline-flex items-center justify-center w-16 h-16">
                          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="2"
                            />
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="#7c3aed"
                              strokeWidth="2"
                              strokeDasharray={`${selectedReport.summary["ndcg@5"] * 100}, 100`}
                            />
                          </svg>
                          <span className="absolute text-sm font-bold text-purple-600">
                            {(selectedReport.summary["ndcg@5"] * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">NDCG@5</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Predictions Section */}
                {selectedReport?.predictions_data && (
                  <div>
                    {/* Filters */}
                    <div className="bg-gray-50 rounded-lg p-4 mb-6">
                      <div className="flex flex-col md:flex-row gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Category:</label>
                          <select 
                            value={selectedCategory}
                            onChange={(e) => setSelectedCategory(e.target.value)}
                            className="border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-[var(--unh-blue)] focus:border-transparent"
                          >
                            <option value="all">All Categories</option>
                            <option value="AS">Academic Standards (AS)</option>
                            <option value="DR">Degree Requirements (DR)</option>
                            <option value="GR">Grading (GR)</option>
                            <option value="GA">Graduation (GA)</option>
                          </select>
                        </div>
                        <div className="flex-1">
                          <label className="block text-sm font-medium text-gray-700 mb-2">Search Questions:</label>
                          <input
                            type="text"
                            placeholder="Search in questions or answers..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-[var(--unh-blue)] focus:border-transparent"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Predictions List */}
                    <div className="space-y-6">
                      {selectedReport.predictions_data.predictions
                          ?.filter((pred: any) => {
                            const matchesCategory = selectedCategory === 'all' || pred.category === selectedCategory;
                            const matchesSearch = searchTerm === '' || 
                              pred.query.toLowerCase().includes(searchTerm.toLowerCase()) ||
                              pred.model_answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
                              pred.reference_answer.toLowerCase().includes(searchTerm.toLowerCase());
                            return matchesCategory && matchesSearch;
                          })
                          .map((pred: any, index: number) => (
                            <div key={pred.id} className="border border-gray-200 rounded-lg p-6">
                              <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center gap-3">
                                  <span className={`px-2 py-1 rounded text-sm font-bold ${
                                    pred.category === 'AS' ? 'bg-blue-100 text-blue-800' :
                                    pred.category === 'DR' ? 'bg-green-100 text-green-800' :
                                    pred.category === 'GR' ? 'bg-purple-100 text-purple-800' :
                                    'bg-orange-100 text-orange-800'
                                  }`}>
                                    {pred.id}
                                  </span>
                                  <span className="text-sm text-gray-500">#{index + 1}</span>
                                </div>
                                <a 
                                  href={pred.url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-[var(--unh-blue)] hover:underline text-sm"
                                >
                                  View Source
                                </a>
                              </div>
                              
                              <div className="mb-4">
                                <h3 className="font-semibold text-lg text-gray-800 mb-2">Question:</h3>
                                <p className="text-gray-700">{pred.query}</p>
                              </div>

                              <div className="grid md:grid-cols-2 gap-6">
                                <div>
                                  <h4 className="font-semibold text-gray-800 mb-2">Model Answer:</h4>
                                  <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                                    <p className="text-gray-800">{pred.model_answer}</p>
                                  </div>
                                </div>
                                
                                <div>
                                  <h4 className="font-semibold text-gray-800 mb-2">Reference Answer:</h4>
                                  <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded">
                                    <p className="text-gray-800">{pred.reference_answer}</p>
                                  </div>
                                </div>
                              </div>

                              <div className="mt-4">
                                <h4 className="font-semibold text-gray-800 mb-2">Retrieved Documents:</h4>
                                <div className="flex flex-wrap gap-2">
                                  {pred.retrieved_ids.map((docId: string, idx: number) => (
                                    <span key={idx} className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-sm">
                                      {docId}
                                    </span>
                                  ))}
                                </div>
                              </div>

                              {pred.nuggets && pred.nuggets.length > 0 && (
                                <div className="mt-4">
                                  <h4 className="font-semibold text-gray-800 mb-2">Key Points:</h4>
                                  <ul className="list-disc list-inside text-sm text-gray-700">
                                    {pred.nuggets.map((nugget: string, idx: number) => (
                                      <li key={idx}>{nugget}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}

                              {/* Individual Question Metrics */}
                              {pred.metrics && (
                                <div className="mt-6">
                                  <h4 className="font-semibold text-gray-800 mb-4">Individual Question Metrics</h4>
                                  
                                  {/* BERTscore Summary Cards */}
                                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                                    <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                                      <h5 className="text-sm font-semibold text-gray-700 mb-2">BERTscore F1</h5>
                                      <p className="text-2xl font-bold text-blue-600">
                                        {(pred.metrics.bertscore_f1 * 100).toFixed(1)}%
                                      </p>
                                      <p className="text-xs text-gray-500 mt-1">Semantic similarity</p>
                                    </div>
                                    
                                    <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                                      <h5 className="text-sm font-semibold text-gray-700 mb-2">SBERT Cosine</h5>
                                      <p className="text-2xl font-bold text-green-600">
                                        {(pred.metrics.sbert_cosine * 100).toFixed(1)}%
                                      </p>
                                      <p className="text-xs text-gray-500 mt-1">Sentence similarity</p>
                                    </div>
                                    
                                    <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                                      <h5 className="text-sm font-semibold text-gray-700 mb-2">Recall@1</h5>
                                      <p className="text-2xl font-bold text-purple-600">
                                        {(pred.metrics["recall@1"] * 100).toFixed(1)}%
                                      </p>
                                      <p className="text-xs text-gray-500 mt-1">Top result accuracy</p>
                                    </div>
                                    
                                    <div className="bg-gray-50 rounded-lg shadow-sm p-4">
                                      <h5 className="text-sm font-semibold text-gray-700 mb-2">NDCG@3</h5>
                                      <p className="text-2xl font-bold text-orange-600">
                                        {(pred.metrics["ndcg@3"] * 100).toFixed(1)}%
                                      </p>
                                      <p className="text-xs text-gray-500 mt-1">Ranking quality</p>
                                    </div>
                                  </div>

                                  {/* Detailed Metrics with Circular Progress */}
                                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
                                    {/* Nugget Metrics */}
                                    <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
                                      <h5 className="text-lg font-bold text-gray-800 mb-4">Nugget Metrics</h5>
                                      <div className="flex justify-around items-center">
                                        {/* Precision Circle */}
                                        <div className="text-center">
                                          <div className="relative inline-flex items-center justify-center w-16 h-16">
                                            <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#e5e7eb"
                                                strokeWidth="2"
                                              />
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#059669"
                                                strokeWidth="2"
                                                strokeDasharray={`${pred.metrics.nugget_precision * 100}, 100`}
                                              />
                                            </svg>
                                            <span className="absolute text-sm font-bold text-green-600">
                                              {(pred.metrics.nugget_precision * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">Precision</div>
                                        </div>
                                        
                                        {/* Recall Circle */}
                                        <div className="text-center">
                                          <div className="relative inline-flex items-center justify-center w-16 h-16">
                                            <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#e5e7eb"
                                                strokeWidth="2"
                                              />
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#dc2626"
                                                strokeWidth="2"
                                                strokeDasharray={`${pred.metrics.nugget_recall * 100}, 100`}
                                              />
                                            </svg>
                                            <span className="absolute text-sm font-bold text-red-600">
                                              {(pred.metrics.nugget_recall * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">Recall</div>
                                        </div>
                                        
                                        {/* F1 Score Circle */}
                                        <div className="text-center">
                                          <div className="relative inline-flex items-center justify-center w-16 h-16">
                                            <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#e5e7eb"
                                                strokeWidth="2"
                                              />
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#2563eb"
                                                strokeWidth="2"
                                                strokeDasharray={`${pred.metrics.nugget_f1 * 100}, 100`}
                                              />
                                            </svg>
                                            <span className="absolute text-sm font-bold text-blue-600">
                                              {(pred.metrics.nugget_f1 * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">F1 Score</div>
                                        </div>
                                      </div>
                                    </div>

                                    {/* Ranking Metrics */}
                                    <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
                                      <h5 className="text-lg font-bold text-gray-800 mb-4">Ranking Performance</h5>
                                      <div className="flex justify-around items-center">
                                        {/* Recall@3 Circle */}
                                        <div className="text-center">
                                          <div className="relative inline-flex items-center justify-center w-16 h-16">
                                            <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#e5e7eb"
                                                strokeWidth="2"
                                              />
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#059669"
                                                strokeWidth="2"
                                                strokeDasharray={`${pred.metrics["recall@3"] * 100}, 100`}
                                              />
                                            </svg>
                                            <span className="absolute text-sm font-bold text-green-600">
                                              {(pred.metrics["recall@3"] * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">Recall@3</div>
                                        </div>
                                        
                                        {/* Recall@5 Circle */}
                                        <div className="text-center">
                                          <div className="relative inline-flex items-center justify-center w-16 h-16">
                                            <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#e5e7eb"
                                                strokeWidth="2"
                                              />
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#059669"
                                                strokeWidth="2"
                                                strokeDasharray={`${pred.metrics["recall@5"] * 100}, 100`}
                                              />
                                            </svg>
                                            <span className="absolute text-sm font-bold text-green-600">
                                              {(pred.metrics["recall@5"] * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">Recall@5</div>
                                        </div>
                                        
                                        {/* NDCG@5 Circle */}
                                        <div className="text-center">
                                          <div className="relative inline-flex items-center justify-center w-16 h-16">
                                            <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#e5e7eb"
                                                strokeWidth="2"
                                              />
                                              <path
                                                d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                                                fill="none"
                                                stroke="#7c3aed"
                                                strokeWidth="2"
                                                strokeDasharray={`${pred.metrics["ndcg@5"] * 100}, 100`}
                                              />
                                            </svg>
                                            <span className="absolute text-sm font-bold text-purple-600">
                                              {(pred.metrics["ndcg@5"] * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">NDCG@5</div>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}