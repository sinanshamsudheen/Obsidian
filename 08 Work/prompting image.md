#!/usr/bin/env python3
"""
Simple AI Image Generation Service - Image + Theme Prompt → Gemini → Result
"""

import os
import base64
import io
import json
import requests
from typing import List, Dict
from PIL import Image


class AIImageGenerationService:
    """Simple service: image + theme prompt → Gemini → result"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.available = bool(self.gemini_api_key)
        
        # Theme prompts
        self.theme_prompts = {
            "modern": (
                "Generate a clean, minimal design food promotional image. "
                "Focus on simplicity, neutral background colors (white, gray, or soft tones), "
                "sharp lighting, and a professional magazine-like aesthetic. "
                "Highlight the dish as the main subject with subtle modern graphic elements "
                "like thin lines, geometric accents, or minimal typography."
            ),
            "vintage": (
                "Generate a nostalgic, classic-style food promotional image. "
                "Use warm, faded tones like sepia or muted pastels to give a retro photography look. "
                "Incorporate vintage textures such as old paper, wooden tables, or rustic plates. "
                "Add decorative elements inspired by the 1950s–70s era, like hand-drawn typography "
                "or classic borders, while keeping the dish as the centerpiece."
            ),
            "neon": (
                "Generate a bold, vibrant promotional image of the dish with neon-inspired aesthetics. "
                "Use glowing neon lights, futuristic backdrops, and bright, saturated colors "
                "such as electric blue, pink, and purple. "
                "The dish should stand out dramatically with high contrast, illuminated by colorful light accents. "
                "Add subtle neon sign-style text or glowing graphic shapes for a modern nightlife vibe."
            ),
            "rustic": (
                "Generate a warm, natural-style food promotional image. "
                "Use earthy tones, wooden textures, and soft natural lighting. "
                "The dish should appear cozy and comforting, styled on rustic surfaces like farm tables "
                "or stoneware plates. "
                "Incorporate elements such as herbs, grains, or fabric textures to emphasize an organic, homely feeling."
            ),
            "festive": (
                "Generate a celebration-themed food promotional image. "
                "Make the dish look vibrant and joyful with bright, colorful decorations, "
                "confetti, ribbons, or sparkling effects. "
                "Use lighting that feels cheerful and energetic. "
                "Incorporate a party-like atmosphere with festive table settings, balloons, "
                "or glowing accents that convey happiness and celebration."
            ),
            "elegant": (
                "Generate a luxurious, sophisticated food promotional image. "
                "Use refined color palettes like black, gold, silver, or deep jewel tones. "
                "Apply soft spotlighting to highlight the dish as the star attraction. "
                "Incorporate upscale tableware such as crystal glasses, fine china, or marble textures. "
                "Keep the design polished and stylish with minimal but tasteful graphic accents."
            )
        }
    
    def generate_images(self, source_image_base64: str, theme: str, count: int = 3) -> List[Dict[str, str]]:
        """
        Simple flow: image + theme prompt → Gemini → result
        """
        if not self.available:
            print("Gemini API key not configured")
            return []
        
        try:
            # Get theme prompt
            theme_lower = theme.lower()
            theme_prompt = self.theme_prompts.get(theme_lower, self.theme_prompts["modern"])
            
            generated_images = []
            
            # Generate images with Gemini
            for i in range(count):
                try:
                    result = self._call_gemini_image_generation(source_image_base64, theme_prompt, theme_lower, i+1)
                    if result:
                        generated_images.append({
                            "filename": f"gemini_{theme_lower}_{i+1}.jpg",
                            "content": result["content"],
                            "content_type": "image/jpeg"
                        })
                except Exception as e:
                    print(f"Error generating image {i+1}: {e}")
            return generated_images
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return []
    
    def _call_gemini_image_generation(self, source_image_base64: str, theme_prompt: str, theme: str, index: int) -> Dict[str, str]:
        """Call Gemini image generation model with proper formatting"""
        try:
            # Prepare the request
            model = "gemini-2.5-flash-image-preview"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
            
            # Format the request body according to Gemini's expected structure
            request_body = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": source_image_base64
                                }
                            },
                            {
                                "text": f"Transform this food image according to this description: {theme_prompt}"
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE", "TEXT"],
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95
                }
            }
            
            print(f"Calling Gemini image generation model for {theme} image #{index}...")
            
            # Send the request
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=request_body
            )
            
            print(f"Gemini response received for {theme} image #{index}")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                # Extract generated image from response
                if 'candidates' in response_data:
                    candidates = response_data['candidates']
                    
                    for candidate in candidates:
                        if 'content' in candidate:
                            content = candidate['content']
                            
                            if 'parts' in content:
                                parts = content['parts']
                                
                                for part in parts:
                                    # Check for inlineData (camelCase in API response)
                                    if 'inlineData' in part:
                                        inline_data = part['inlineData']
                                        
                                        # Check if we have actual image data
                                        if 'data' in inline_data:
                                            # Return the generated image data as base64
                                            # The data is already base64 encoded from the API
                                            return {
                                                "content": inline_data['data'],
                                                "content_type": "image/jpeg"
                                            }
            
            print(f"No image data in Gemini response for {theme} #{index}")
            return None
                
        except Exception as e:
            print(f"Gemini call failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Global instance
ai_image_service = AIImageGenerationService() 


#!/usr/bin/env python3
"""
Simple AI Image Generation Service - Image + Theme Prompt → Gemini → Result
"""

import os
import base64
import io
import json
import requests
from typing import List, Dict
from PIL import Image


class AIImageGenerationService:
    """Simple service: image + theme prompt → Gemini → result"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.available = bool(self.gemini_api_key)
        
        # Theme prompts
        self.theme_prompts = {
            "modern": (
                "Generate a clean, minimal design food promotional image. "
                "Focus on simplicity, neutral background colors (white, gray, or soft tones), "
                "sharp lighting, and a professional magazine-like aesthetic. "
                "Highlight the dish as the main subject with subtle modern graphic elements "
                "like thin lines, geometric accents, or minimal typography."
            ),
            "vintage": (
                "Generate a nostalgic, classic-style food promotional image. "
                "Use warm, faded tones like sepia or muted pastels to give a retro photography look. "
                "Incorporate vintage textures such as old paper, wooden tables, or rustic plates. "
                "Add decorative elements inspired by the 1950s–70s era, like hand-drawn typography "
                "or classic borders, while keeping the dish as the centerpiece."
            ),
            "neon": (
                "Generate a bold, vibrant promotional image of the dish with neon-inspired aesthetics. "
                "Use glowing neon lights, futuristic backdrops, and bright, saturated colors "
                "such as electric blue, pink, and purple. "
                "The dish should stand out dramatically with high contrast, illuminated by colorful light accents. "
                "Add subtle neon sign-style text or glowing graphic shapes for a modern nightlife vibe."
            ),
            "rustic": (
                "Generate a warm, natural-style food promotional image. "
                "Use earthy tones, wooden textures, and soft natural lighting. "
                "The dish should appear cozy and comforting, styled on rustic surfaces like farm tables "
                "or stoneware plates. "
                "Incorporate elements such as herbs, grains, or fabric textures to emphasize an organic, homely feeling."
            ),
            "festive": (
                "Generate a celebration-themed food promotional image. "
                "Make the dish look vibrant and joyful with bright, colorful decorations, "
                "confetti, ribbons, or sparkling effects. "
                "Use lighting that feels cheerful and energetic. "
                "Incorporate a party-like atmosphere with festive table settings, balloons, "
                "or glowing accents that convey happiness and celebration."
            ),
            "elegant": (
                "Generate a luxurious, sophisticated food promotional image. "
                "Use refined color palettes like black, gold, silver, or deep jewel tones. "
                "Apply soft spotlighting to highlight the dish as the star attraction. "
                "Incorporate upscale tableware such as crystal glasses, fine china, or marble textures. "
                "Keep the design polished and stylish with minimal but tasteful graphic accents."
            )
        }
    
    def generate_images(self, source_image_base64: str, theme: str, count: int = 3) -> List[Dict[str, str]]:
        """
        Simple flow: image + theme prompt → Gemini → result
        """
        if not self.available:
            print("Gemini API key not configured")
            return []
        
        try:
            # Get theme prompt
            theme_lower = theme.lower()
            theme_prompt = self.theme_prompts.get(theme_lower, self.theme_prompts["modern"])
            
            generated_images = []
            
            # Generate images with Gemini
            for i in range(count):
                try:
                    result = self._call_gemini_image_generation(source_image_base64, theme_prompt, theme_lower, i+1)
                    if result:
                        generated_images.append({
                            "filename": f"gemini_{theme_lower}_{i+1}.jpg",
                            "content": result["content"],
                            "content_type": "image/jpeg"
                        })
                except Exception as e:
                    print(f"Error generating image {i+1}: {e}")
            return generated_images
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return []
    
    def _call_gemini_image_generation(self, source_image_base64: str, theme_prompt: str, theme: str, index: int) -> Dict[str, str]:
        """Call Gemini image generation model with proper formatting"""
        try:
            # Prepare the request
            model = "gemini-2.5-flash-image-preview"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
            
            # Format the request body according to Gemini's expected structure
            request_body = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": source_image_base64
                                }
                            },
                            {
                                "text": f"Transform this food image according to this description: {theme_prompt}"
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE", "TEXT"],
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95
                }
            }
            
            print(f"Calling Gemini image generation model for {theme} image #{index}...")
            
            # Send the request
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=request_body
            )
            
            print(f"Gemini response received for {theme} image #{index}")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                # Extract generated image from response
                if 'candidates' in response_data:
                    candidates = response_data['candidates']
                    
                    for candidate in candidates:
                        if 'content' in candidate:
                            content = candidate['content']
                            
                            if 'parts' in content:
                                parts = content['parts']
                                
                                for part in parts:
                                    # Check for inlineData (camelCase in API response)
                                    if 'inlineData' in part:
                                        inline_data = part['inlineData']
                                        
                                        # Check if we have actual image data
                                        if 'data' in inline_data:
                                            # Return the generated image data as base64
                                            # The data is already base64 encoded from the API
                                            return {
                                                "content": inline_data['data'],
                                                "content_type": "image/jpeg"
                                            }
            
            print(f"No image data in Gemini response for {theme} #{index}")
            return None
                
        except Exception as e:
            print(f"Gemini call failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Global instance
ai_image_service = AIImageGenerationService() 



