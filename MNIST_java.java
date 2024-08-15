// Video lecture followed : https://www.youtube.com/@raeisonline7254

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MNIST_java {
    static final String train_path = "utility/mnist_train.csv";
    static final String test_path = "utility/mnist_test.csv";

    static class Image {
        double[][] data;
        int label;

        public Image(double[][] data, int label) {
            this.data = data;
            this.label = label;
        }

        public double[][] getData() {
            return data;
        }

        public int getLabel() {
            return label;
        }
        @Override
        public String toString(){
            StringBuilder s= new StringBuilder();
            for(double[] x:data){
                for(double y:x){
                    s.append(y).append(", ");
                }
                s.append("\n");
            }
            return s.toString();
        }
    }
    static class DataReader {
        private final int rows = 28;
        private final int cols = 28;

        public List<Image> readData(String path){
            List<Image> images = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(path))) {
                String line;
                while((line = br.readLine()) !=null){
                    String[] pixels = line.split(",");
                    double[][] data = new double[rows][cols];
                    int label = Integer.parseInt(pixels[0]);

                    int i = 1;

                    for(int x=0; x<rows; x++){
                        for(int y=0; y<cols; y++){
                            data[x][y] = (double)Integer.parseInt(pixels[i]);
                            i++;
                        }
                    }
                    images.add(new Image(data, label));
                }
            } catch(Exception e) {
                System.out.println(e);
            }
            return images;
        }
    } // Load data to class Image object

    abstract class Layer {
        Layer next;

        public Layer getNext() {
            return next;
        }

        public void setNext(Layer next) {
            this.next = next;
        }

        public Layer getPrev() {
            return prev;
        }

        public void setPrev(Layer prev) {
            this.prev = prev;
        }

        Layer prev;
        abstract int getOutputLength();
        abstract int getOutputRows();
        abstract int getOutputCols();
        abstract int getOutputElements();

        abstract double[] getOutput(List<double[][]> input);
        abstract double[] getOutput(double[] input);
        abstract void backProp(List<double[][]> input);
        abstract void backProp(double[] input);

        double[] flattenArray(List<double[][]> input){
            int length = input.size();
            int rows = input.get(0).length;
            int cols = input.get(0)[0].length;

            double[] ar = new double[length + rows + cols];
            int i=0;
            for(int l=0;l<length;l++) {
                for (int x = 0; x < rows; x++) {
                    for (int y = 0; y < cols; y++) {
                        ar[i] = input.get(l)[x][y];
                        i++;
                    }
                }
            }
            return ar;
        }
        List<double[][]> getMatrix(double[] input, int length, int rows, int cols){
            List<double[][]> out = new ArrayList<>();
            int i=0;
            for(int l=0;l<length;l++){
                double[][] matrix = new double[rows][cols];
                for(int x=0;x<rows;x++){
                    for(int y=0;y<cols;y++){
                        matrix[x][y] = input[i++];
                    }
                }
                out.add(matrix);
            }
            return out;
        }
    } // To be used and further to implement linear layer, max-pool layer

    class LinearLayer extends Layer{
        private double[][] wt;
        private int in_channel;
        private int out_channel;
        private long SEED;
        private double[] prev_value;
        private double learn_rate;

        public LinearLayer(int in_channel, int out_channel, long SEED, double learn_rate) {
            this.wt = new double[in_channel][out_channel];
            this.in_channel = in_channel;
            this.out_channel = out_channel;
            this.SEED = SEED;
            setRandomWeights();
        }
        public void setRandomWeights(){
            Random random = new Random(SEED);
            for(int x=0;x<in_channel;x++){
                for(int y=0;y<out_channel;y++){
                    wt[x][y] = random.nextGaussian();
                }
            }
        }
        public double leaky_reLU(double x){
            return x>0? x:0;
        }
        public double d_reLU(double x){
            return x>0? 1:0.001;
        }
        public double[] forward(double[] x){
            prev_value = x;
            double[] y = new double[out_channel];

            for(int i=0;i<in_channel;i++){
                for(int j=0;j<out_channel;j++){
                    y[j] += x[i] * wt[i][j];
                }
            }
            for(int i=0;i<in_channel;i++){
                for(int j=0;j<out_channel;j++){
                    y[j] = leaky_reLU(y[j]);
                }
            }
            return y;
        }

        @Override
        void backProp(List<double[][]> input) {
            backProp(flattenArray(input));
        }

        @Override
        void backProp(double[] d_loss_out) {
            // z = w*x
            // out = leaky_reLU(z)
            double[] d_loss_x = new double[in_channel];
            double d_out_z;
            double d_z_w;
            double d_loss_w;
            double d_z_x;

            for(int x=0; x<in_channel;x++){
                double d_loss_x_sum = 0;
                for(int y=0;y<out_channel;y++){
                    d_out_z = d_reLU(prev_value[y]);
                    d_z_w = prev_value[x];
                    d_z_x = wt[x][y];

                    d_loss_w = d_loss_out[y] * d_out_z * d_z_w;
                    wt[x][y] = wt[x][y] - d_loss_w * learn_rate;

                    d_loss_x_sum += d_loss_out[x] * d_out_z * d_z_x;
                    // Error for previous layer
                }
                d_loss_x[x] = d_loss_x_sum;
            }
            if(prev!=null)
                prev.backProp(d_loss_x);
        }

        @Override
        int getOutputLength() {
            return 0;
        }

        @Override
        int getOutputRows() {
            return 0;
        }

        @Override
        int getOutputCols() {
            return 0;
        }

        @Override
        int getOutputElements() {
            return out_channel;
        }

        @Override
        double[] getOutput(List<double[][]> input) {
            return getOutput(flattenArray(input));
        }

        @Override
        double[] getOutput(double[] input) {
            double[] output = forward(input);
            if(next !=null)
                return next.getOutput(output);
            return output;
        }
    }

    class MaxpoolLayer extends Layer{
        @Override
        int getOutputLength() {
            return 0;
        }

        @Override
        int getOutputRows() {
            return 0;
        }

        @Override
        int getOutputCols() {
            return 0;
        }

        @Override
        int getOutputElements() {
            return 0;
        }

        @Override
        double[] getOutput(List<double[][]> input) {
            return new double[0];
        }

        @Override
        double[] getOutput(double[] input) {
            return new double[0];
        }

        @Override
        void backProp(List<double[][]> input) {

        }

        @Override
        void backProp(double[] input) {

        }
    }
    public static void main(String[] args){
        List<Image> images = new DataReader().readData(test_path);
        System.out.println(images.get(0).toString());
    }
}
