package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.StyleSpan;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.bumptech.glide.Glide;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabeling;
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE = 1;
    private static final int REQUEST_STORAGE_PERMISSION = 100;
    private static final float THRESHOLD = -0.2f;

    private ImageView imageView;
    private TextView textViewResult;
    private Interpreter tfliteInterpreter;
    private List<String> flowerLabels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        Button buttonUpload = findViewById(R.id.buttonUpload);
        textViewResult = findViewById(R.id.textViewResult);

        // Load TFLite model and labels
        try {
            tfliteInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "flower_model.tflite"));
            flowerLabels = FileUtil.loadLabels(this, "labels.txt");
        } catch (IOException e) {
            Log.e("ModelError", "Error loading model or labels: " + e.getMessage());
            textViewResult.setText("Error loading model or labels.");
        }

        buttonUpload.setOnClickListener(view -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        REQUEST_STORAGE_PERMISSION);
            } else {
                openGallery();
            }
        });
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openGallery();
        } else {
            textViewResult.setText("Permission denied.");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            if (imageUri != null) {
                new ImageProcessingTask().execute(imageUri);
            } else {
                textViewResult.setText("Error: Image selection failed.");
            }
        }
    }

    private class ImageProcessingTask extends AsyncTask<Uri, Void, Bitmap> {
        @Override
        protected Bitmap doInBackground(Uri... uris) {
            try {
                return Glide.with(MainActivity.this)
                        .asBitmap()
                        .load(uris[0])
                        .override(224, 224)
                        .submit()
                        .get();
            } catch (ExecutionException | InterruptedException | OutOfMemoryError e) {
                Log.e("ImageError", "Error loading image: " + e.getMessage());
                return null;
            }
        }

        @Override
        protected void onPostExecute(Bitmap bitmap) {
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap);
                analyzeImageByRegions(bitmap);
            } else {
                textViewResult.setText("Error loading image.");
            }
        }
    }

    private void analyzeImageByRegions(Bitmap bitmap) {
        // Define regions of interest
        Rect regionForCustomModel = new Rect(0, 0, bitmap.getWidth() / 2, bitmap.getHeight());
        Rect regionForMLKit = new Rect(bitmap.getWidth() / 2, 0, bitmap.getWidth(), bitmap.getHeight());

        Bitmap customModelRegion = cropRegion(bitmap, regionForCustomModel);
        Bitmap mlKitRegion = cropRegion(bitmap, regionForMLKit);

        SpannableStringBuilder styledResults = new SpannableStringBuilder(); // Create a SpannableStringBuilder to combine both texts

        // Process the custom model region
        if (customModelRegion != null) {
            // Create a SpannableString for "Main Prediction:" and make it bold
            SpannableString boldTextMain = new SpannableString("Main Prediction:\n");
            boldTextMain.setSpan(new StyleSpan(Typeface.BOLD), 0, boldTextMain.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
            styledResults.append(boldTextMain); // Append the bold "Main Prediction" text
            styledResults.append(predictWithCustomModel(customModelRegion)).append("\n"); // Append the result of predictWithCustomModel
        }

        // Process the ML Kit region
        if (mlKitRegion != null) {
            predictWithMLKit(mlKitRegion, styledResults);
        }

        // Set the final styled text to your TextView
        textViewResult.setText(styledResults); // Use the SpannableStringBuilder for text display
    }


    private Bitmap cropRegion(Bitmap source, Rect region) {
        return Bitmap.createBitmap(source, region.left, region.top, region.width(), region.height());
    }

    private String predictWithCustomModel(Bitmap bitmap) {
        if (tfliteInterpreter == null || flowerLabels == null) {
            return "Model or labels not loaded properly.";
        }

        try {
            ByteBuffer inputBuffer = preprocessImage(bitmap);
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, flowerLabels.size()}, DataType.FLOAT32);
            tfliteInterpreter.run(inputBuffer, outputBuffer.getBuffer());

            float[] output = outputBuffer.getFloatArray();
            int maxIndex = 0;
            for (int i = 1; i < output.length; i++) {
                if (output[i] > output[maxIndex]) {
                    maxIndex = i;
                }
            }

            float maxConfidence = output[maxIndex];
            return maxConfidence >= THRESHOLD
                    ? flowerLabels.get(maxIndex) + " (Confidence: " + maxConfidence + ")"
                    : "Custom model did not find a confident match.";
        } catch (Exception e) {
            Log.e("PredictionError", "Error during prediction: " + e.getMessage());
            return "Prediction failed.";
        }
    }

    private void predictWithMLKit(Bitmap bitmap, SpannableStringBuilder styledResults) {
        try {
            InputImage image = InputImage.fromBitmap(bitmap, 0);
            ImageLabeler labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS);

            labeler.process(image)
                    .addOnSuccessListener(labels -> {
                        // Create a SpannableString for "Other Predictions:" and make it bold
                        SpannableString boldTextOther = new SpannableString("Other Predictions:\n");
                        boldTextOther.setSpan(new StyleSpan(Typeface.BOLD), 0, boldTextOther.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                        styledResults.append(boldTextOther); // Append the bold "Other Predictions" text

                        // Append the rest of the prediction results
                        for (ImageLabel label : labels) {
                            styledResults.append(label.getText())
                                    .append(" (Confidence: ")
                                    .append(String.format("%.2f", label.getConfidence()))
                                    .append(")\n");
                        }

                        // Set the final styled text to your TextView
                        textViewResult.setText(styledResults); // Use the SpannableStringBuilder for text display
                    })
                    .addOnFailureListener(e -> {
                        Log.e("MLKitError", "ML Kit prediction failed: " + e.getMessage());
                        styledResults.append("ML Kit prediction failed.");
                        textViewResult.setText(styledResults.toString());
                    });
        } catch (Exception e) {
            Log.e("MLKitError", "Error creating InputImage: " + e.getMessage());
            textViewResult.setText("Error processing image with ML Kit.");
        }
    }


    private ByteBuffer preprocessImage(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }

        return byteBuffer;
    }
}