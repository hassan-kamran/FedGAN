import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { trainingAPI } from '../services/api';

export default function ImageGallery() {
  const { id } = useParams();
  const [images, setImages] = useState([]);

  useEffect(() => {
    loadImages();
  }, [id]);

  const loadImages = async () => {
    try {
      const response = await trainingAPI.getImages(id, { page: 1, page_size: 50 });
      setImages(response.data.images);
    } catch (error) {
      console.error('Error loading images:', error);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Synthetic Images</h2>
      {images.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500">No images generated yet</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {images.map((image) => (
            <div key={image.id} className="card p-2">
              <div className="aspect-square bg-gray-200 rounded-lg mb-2"></div>
              <p className="text-xs text-gray-600 truncate">{image.filename}</p>
              <p className="text-xs text-gray-500">Round {image.generation_round}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
