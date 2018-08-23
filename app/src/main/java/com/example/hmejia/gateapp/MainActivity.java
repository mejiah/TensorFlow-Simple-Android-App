package com.example.hmejia.gateapp;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    private static final String MODEL_FILE = "file:///android_asset/frozen_results.pb";

    private static final String[] INPUT_NODES = {"modelInputA","modelInputB"};
    private static final String[] OUTPUT_NODES = {"modelOutput"};
    private static final int[] INPUT_DIM = {1};

    private TensorFlowInferenceInterface inferenceInterface;

    public void computeNumbers(View view) {

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        final EditText etInputA = (EditText) findViewById(R.id.num1);
        final EditText etInputB = (EditText) findViewById(R.id.num2);

        float numA = Float.parseFloat(etInputA.getText().toString());
        float numB = Float.parseFloat(etInputB.getText().toString());

        float[] modelInputA = {numA};
        float[] modelInputB = {numB};

        inferenceInterface.feed(INPUT_NODES[0], modelInputA, INPUT_DIM[0]);
        inferenceInterface.feed(INPUT_NODES[1], modelInputB, INPUT_DIM[0]);

        inferenceInterface.run(OUTPUT_NODES);

        float[] modelOutput = new float[1];
        inferenceInterface.fetch(OUTPUT_NODES[0], modelOutput);

        final EditText textViewR = (EditText) findViewById(R.id.num3);
        textViewR.setText(Float.toString(modelOutput[0]) );
    }

}
