if __name__ == '__main__':
    
    skintone_score = summed_score - windshield_tint_score - shadow_score
    
    print(skintone_score)
    
    ####
    
    # Adjust for windshield shade (e.g., normalize skin tone prediction)
    adjusted_skin_tone = skin_tone_model.predict(grayscale_array)[0] - windshield_shade

    print(f »Predicted Driver Skin Tone: {adjusted_skin_tone} »)
    return adjusted_skin_tone

# Example usage
predicted_skin_tone = predict_driver_skin_tone(grayscale_image, skin_tone_model, windshield_shade_model)