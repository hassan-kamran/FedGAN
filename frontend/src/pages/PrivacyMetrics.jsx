import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { trainingAPI } from '../services/api';

export default function PrivacyMetrics() {
  const { id } = useParams();
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    loadMetrics();
  }, [id]);

  const loadMetrics = async () => {
    try {
      const response = await trainingAPI.getPrivacyMetrics(id);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    }
  };

  if (!metrics) {
    return <div className="flex justify-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div></div>;
  }

  const getRiskColor = (score) => {
    if (score < 30) return 'text-green-600';
    if (score < 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Privacy Evaluation Metrics</h2>

      <div className="card">
        <h3 className="text-xl font-semibold mb-4">Overall Privacy Risk Score</h3>
        <div className="text-center">
          <div className={`text-6xl font-bold ${getRiskColor(metrics.privacy_risk_score)}`}>
            {metrics.privacy_risk_score?.toFixed(1) || 'N/A'} / 100
          </div>
          <p className="text-gray-600 mt-2">
            {metrics.privacy_risk_score < 30 ? 'Low Risk' : metrics.privacy_risk_score < 70 ? 'Medium Risk' : 'High Risk'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold mb-2">Membership Inference Risk</h3>
          <p className="text-3xl font-bold text-primary-600">{(metrics.membership_inference_risk * 100).toFixed(1)}%</p>
          <p className="text-sm text-gray-600 mt-2">Risk of identifying training data membership</p>
        </div>

        <div className="card">
          <h3 className="font-semibold mb-2">Model Inversion Risk</h3>
          <p className="text-3xl font-bold text-medical-600">{(metrics.model_inversion_risk * 100).toFixed(1)}%</p>
          <p className="text-sm text-gray-600 mt-2">Risk of reconstructing training data</p>
        </div>

        <div className="card">
          <h3 className="font-semibold mb-2">Differential Privacy (ε)</h3>
          <p className="text-3xl font-bold text-purple-600">{metrics.epsilon?.toFixed(2) || 'N/A'}</p>
          <p className="text-sm text-gray-600 mt-2">Privacy budget (lower is better)</p>
        </div>

        <div className="card">
          <h3 className="font-semibold mb-2">Attack Success Rate</h3>
          <p className="text-3xl font-bold text-orange-600">{(metrics.attack_success_rate * 100).toFixed(1)}%</p>
          <p className="text-sm text-gray-600 mt-2">Success rate of privacy attacks</p>
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-4">Recommendations</h3>
        <ul className="space-y-2">
          {metrics.privacy_risk_score > 70 && (
            <li className="flex items-start">
              <span className="text-red-500 mr-2">⚠️</span>
              <span>High privacy risk detected. Consider increasing the number of federated clients or adding more noise.</span>
            </li>
          )}
          {metrics.epsilon > 10 && (
            <li className="flex items-start">
              <span className="text-yellow-500 mr-2">⚠️</span>
              <span>Epsilon value is high. Consider implementing differential privacy mechanisms.</span>
            </li>
          )}
          {metrics.privacy_risk_score < 30 && (
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Good privacy guarantees. The federated learning setup provides adequate protection.</span>
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}
