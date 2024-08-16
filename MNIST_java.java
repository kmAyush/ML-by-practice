/*
Acknowledgement :-
https://www.youtube.com/@raeisonline7254

--------------OUTPUT----------------
Dataset size :
Train : 60000
Test : 10000
Accuracy without training : 0.1223
Accuracy after 1 epoch : 0.6065
Accuracy after 2 epoch : 0.7292
Accuracy after 3 epoch : 0.7842
Accuracy after 4 epoch : 0.902
Accuracy after 5 epoch : 0.9076
Accuracy after 6 epoch : 0.9121
Accuracy after 7 epoch : 0.9149
Accuracy after 8 epoch : 0.9176
Accuracy after 9 epoch : 0.9187
Accuracy after 10 epoch : 0.9191
-----------------------------------
*/
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.util.Collections.shuffle;

public class MNIST_java {
    static final String train_path = "data/mnist_train.csv";
    static final String test_path = "data/mnist_test.csv";

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
        Layer prev;

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

            double[] ar = new double[length * rows * cols];
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
        private double learn_rate;

        private double[] prev_x;
        private double[] prev_z;

        public LinearLayer(int in_channel, int out_channel, long SEED, double learn_rate) {
            this.wt = new double[in_channel][out_channel];
            this.in_channel = in_channel;
            this.out_channel = out_channel;
            this.SEED = SEED;
            this.learn_rate = learn_rate;
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
            prev_x = x;
            double[] z = new double[out_channel];
            double[] out = new double[out_channel];

            for(int i=0;i<in_channel;i++){
                for(int j=0;j<out_channel;j++){
                    z[j] += x[i] * wt[i][j];
                }
            }
            prev_z = z;
            for(int i=0;i<in_channel;i++){
                for(int j=0;j<out_channel;j++){
                    out[j] = leaky_reLU(z[j]);
                }
            }
            return out;
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
                    d_out_z = d_reLU(prev_z[y]);
                    d_z_w = prev_x[x];
                    d_z_x = wt[x][y];

                    d_loss_w = d_loss_out[y] * d_out_z * d_z_w;
                    wt[x][y] = wt[x][y] - d_loss_w * learn_rate;

                    d_loss_x_sum += d_loss_out[y] * d_out_z * d_z_x;
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

    class MaxPoolLayer extends Layer{
        private int stride;
        private int window;
        private int in_channel;
        private int rows;
        private int cols;
        private List<int[][]> prev_max_row;
        private List<int[][]> prev_max_col;

        public MaxPoolLayer(int stride, int window, int in_channel, int rows, int cols) {
            this.stride = stride;
            this.window = window;
            this.in_channel = in_channel;
            this.rows = rows;
            this.cols = cols;
        }
        private List<double[][]> forward(List<double[][]> input){
            List<double[][]> output = new ArrayList<>();
            prev_max_row = new ArrayList<>();
            prev_max_col = new ArrayList<>();
            for (int x=0;x<input.size();x++){
                output.add(maxPool(input.get(x)));
            }
            return output;
        }
        public double[][] maxPool(double[][] input){
            int pool_rows = getOutputRows();
            int pool_cols = getOutputCols();

            int[][] max_row_index = new int[pool_rows][pool_cols];
            int[][] max_col_index = new int[pool_rows][pool_cols];

            double[][] output =  new double[pool_rows][pool_cols];
            for(int x=0;x<pool_rows;x+=stride){
                for(int y=0;y<pool_cols;y+=stride){
                    double max = 0.0;
                    max_row_index[x][y] = -1;
                    max_col_index[x][y] = -1;
                    for(int i=0;i<window;i++){
                        for(int j=0;j<window;j++){
                            if(max < input[x+i][y+j]) {
                                max=input[x+i][y+j];
                                max_row_index[x][y] = x+i;
                                max_col_index[x][y] = y+j;
                            }
                        }
                    }
                    output[x][y] = max;
                }
            }
            prev_max_row.add(max_row_index);
            prev_max_col.add(max_col_index);
            return output;
        }

        @Override
        int getOutputLength() {
            return in_channel;
        }

        @Override
        int getOutputRows() {
            return (rows-window)/stride+1;
        }

        @Override
        int getOutputCols() {
            return (cols-window)/stride+1;
        }

        @Override
        int getOutputElements() {
            return in_channel*getOutputCols()*getOutputRows();
        }

        @Override
        double[] getOutput(List<double[][]> input) {
            List<double[][]> output = forward(input);
            return next.getOutput(output);
        }

        @Override
        double[] getOutput(double[] input) {
            return getOutput(getMatrix(input, in_channel, rows, cols));
        }

        @Override
        void backProp(List<double[][]> dl_out) {
            List<double[][]> dx_l = new ArrayList<>();
            int l=0;
            int pool_rows = getOutputRows();
            int pool_cols = getOutputCols();
            for (double[][] dl_o_i:dl_out){
                double[][] error = new double[rows][cols];

                for(int x=0;x<pool_rows;x++){
                    for(int y=0;y<pool_cols;y++){
                        int row_idx = prev_max_row.get(l)[x][y];
                        int col_idx = prev_max_col.get(l)[x][y];

                        if(row_idx != -1){
                            error[row_idx][col_idx] += dl_o_i[x][y];
                        }
                    }
                }
                dx_l.add(error);
                l++;
            }
            if(prev!=null) prev.backProp(dx_l);
        }

        @Override
        void backProp(double[] dl_out) {
            backProp(getMatrix(dl_out, getOutputLength(), getOutputRows(), getOutputCols()));
        }
    }

    class ConvolutionalLayer extends Layer{
        private List<double[][]> kernel;
        private int kernel_size;
        private int stride;
        private int in_channel;
        private int rows;
        private int cols;
        private long SEED;
        private double learning_rate;
        private List<double[][]> prev_input;

        public ConvolutionalLayer(int kernel_size, int stride, int in_channel, int rows, int cols, long SEED, int kernel_num,double lr) {
            this.kernel_size = kernel_size;
            this.stride = stride;
            this.in_channel = in_channel;
            this.rows = rows;
            this.cols = cols;
            this.SEED = SEED;
            this.learning_rate = lr;

            generateRandomFilters(kernel_num);
        }

        private void generateRandomFilters(int kernel_num){
            List<double[][]> filters = new ArrayList<>();
            Random random = new Random(SEED);
            for(int i=0;i<kernel_num;i++){
                double[][] new_kernel = new double[kernel_size][kernel_size];
                for(int x=0;x<kernel_size;x++){
                    for(int y=0;y<kernel_size;y++){
                        new_kernel[x][y] = random.nextGaussian();
                    }
                }
                filters.add(new_kernel);
            }
            kernel = filters;
        }

        private List<double[][]> forward(List<double[][]> input){
            List<double[][]> output = new ArrayList<>();
            prev_input = input;
            for(int m=0;m<input.size();m++){
                for(double[][] f : kernel){
                    output.add(convolve(input.get(m), f, stride));
                }
            }
            return output;
        }

        private double[][] convolve(double[][] input, double[][] filter, int stride){
            int in_rows = input.length;
            int in_cols = input[0].length;
            int f_rows = filter.length;
            int f_cols = filter[0].length;
            int out_rows = (in_rows - f_rows) / stride + 1;
            int out_cols = (in_cols - f_cols) / stride + 1;

            double[][] output = new double[out_rows][out_cols];

            int out_row = 0;
            for(int x=0; x<=in_rows-f_rows; x+=stride){
                int out_col=0;
                for(int y=0; y<=in_cols-f_cols; y+=stride){
                    double sum = 0.0;
                    for(int x1=0; x1<f_rows; x1++){
                        for(int y1=0; y1<f_cols; y1++){
                            int row_index = x1+x;
                            int col_index = y1+y;
                            sum += filter[x1][y1]*input[row_index][col_index];
                        }
                    }
                    output[out_row][out_col] = sum;
                    out_col++;
                }
                out_row++;
            }
            return output;
        }

        double[][] spaceArray(double[][] input){
            if(stride == 1) return input;
            int in_rows = input.length;
            int in_cols = input[0].length;
            int out_rows = (in_rows - 1) * stride + 1;
            int out_cols = (in_cols - 1) * stride + 1;

            double[][] output = new double[out_rows][out_cols];

            for(int x=0;x<in_rows;x++){
                for(int y=0;y<in_cols;y++){
                    // Stretching each input by step size
                    output[x*stride][y*stride] = input[x][y];
                }
            }
            return output;
        }
        @Override
        void backProp(List<double[][]> dl_out) {
            List<double[][]> prev_dl_out = new ArrayList<>();
            List<double[][]> d_filters = new ArrayList<>();
            for(int f=0; f<kernel.size();f++){
                d_filters.add(new double[kernel_size][kernel_size]);
            }
            for(int n=0;n<prev_input.size();n++){
                double[][] input_error = new double[rows][cols];
                for(int f=0;f<kernel.size();f++){
                    double[][] current_filter = kernel.get(f);
                    double[][] error = dl_out.get(n*kernel.size() + f);
                    double[][] spaced_error = spaceArray(error);
                    double[][] dl_f = convolve(prev_input.get(n), spaced_error, 1);
                    double[][] delta = Utils.multiply(dl_f, learning_rate*-1);
                    double[][] new_d_filter = Utils.add_matrices(d_filters.get(f), delta);
                    d_filters.set(f, new_d_filter);

                    double[][] flip_error = flip_matrix(spaced_error);
                    input_error = Utils.add_matrices(input_error, full_convolve(current_filter, flip_error));
                }
                prev_dl_out.add(input_error);
            }
            for(int f=0;f<kernel.size();f++){
                double[][] new_filter = Utils.add_matrices(d_filters.get(f),kernel.get(f));
                kernel.set(f, new_filter);
            }
            if(prev !=null)
                prev.backProp(prev_dl_out);
        }

        double[][] flip_matrix(double[][] array){
            int rows = array.length;
            int cols = array[0].length;
            double[][] output = new double[rows][cols];
            for(int x=0;x<rows;x++){ // Flip Horizontally
                System.arraycopy(array[x], 0, output[rows - x - 1], 0, cols);
            }
            array = output;
            output = new double[rows][cols];
            for(int x=0;x<rows;x++){ // Flip Horizontally
                for(int y=0;y<cols;y++){
                    output[x][cols-y-1] = array[x][y];
                }
            }
            return output;
        }

        private double[][] full_convolve(double[][] input, double[][] filter){
            int in_rows = input.length;
            int in_cols = input[0].length;
            int f_rows = filter.length;
            int f_cols = filter[0].length;
            int out_rows = (in_rows + f_rows) + 1;
            int out_cols = (in_cols + f_cols) + 1;

            double[][] output = new double[out_rows][out_cols];

            int out_row = 0;
            for(int x=-f_rows+1 ;x<in_rows;x++){
                int out_col=0;
                for(int y=-f_cols+1; y<in_cols; y++){
                    double sum = 0.0;
                    for(int x1=0;x1<f_rows;x1++){
                        for(int y1=0;y1<f_cols;y1++){
                            int row_index = x1+x;
                            int col_index = y1+y;
                            if(row_index >=0 && row_index< in_rows &&
                               col_index >=0 && col_index< in_cols)
                                sum += (filter[x1][y1]*input[row_index][col_index]);
                        }
                    }
                    output[out_row][out_col] = sum;
                    out_col++;
                }
                out_row++;
            }
            return output;
        }

        @Override
        void backProp(double[] dl_out) {
            List<double[][]> matrix = getMatrix(dl_out, in_channel, rows, cols);
            backProp(matrix);
        }
        @Override
        int getOutputLength() {
            return kernel.size() * in_channel;
        }

        @Override
        int getOutputRows() {
            return (rows - kernel_size) / stride + 1;
        }

        @Override
        int getOutputCols() {
            return (cols - kernel_size) / stride + 1;
        }

        @Override
        int getOutputElements() {
            return getOutputRows() * getOutputCols() * getOutputLength();
        }

        @Override
        double[] getOutput(List<double[][]> input) {
            return next.getOutput(forward(input));
        }

        @Override
        double[] getOutput(double[] input) {
            return getOutput(getMatrix(input, in_channel, rows, cols));
        }
    }

    static class Utils {
        static double[][] add_matrices(double[][] a,double[][] b){
            int m = a.length, n = a[0].length;
            double[][] sum = new double[m][n];
            for(int x=0;x<m;x++){
                for(int y=0;y<n;y++){
                    sum[x][y] = a[x][y] + b[x][y];
                }
            }
            return sum;
        }
        static double[] add_matrices(double[] a,double[] b){
            int m = a.length;
            double[] sum = new double[m];
            for(int x=0;x<m;x++){
                sum[x] = a[x] + b[x];
            }
            return sum;
        }
        static double[][] multiply(double[][] a, double n){
            double[][] out = new double[a.length][a[0].length];
            for(int x=0;x<a.length;x++)
                for(int y=0;y<a.length;y++)
                    out[x][y] = a[x][y]*n;
            return out;
        }
        static double[] multiply(double[] a, double n){
            double[] out = new double[a.length];
            for(int x=0;x<a.length;x++)
                out[x] = a[x]*n;
            return out;
        }
    }

    class NeuralNet {
        List<Layer> layers;
        double scale_factor;

        public NeuralNet(List<Layer> layers, double scale_factor) {
            this.layers = layers;
            this.scale_factor = scale_factor;
            link_layers();
        }
        private void link_layers(){
            if(layers.size()<=1) return;

            for(int i=0;i<layers.size();i++){
                if(i==0){
                    layers.get(i).setNext(layers.get(i+1));
                } else if(i==layers.size()-1) {
                    layers.get(i).setPrev(layers.get(i - 1));
                } else {
                    layers.get(i).setPrev(layers.get(i - 1));
                    layers.get(i).setNext(layers.get(i+1));
                }
            }
        }
        public double[] getErrors(double[] neural_output, int correct_ans){
            int num_classes = neural_output.length;
            double[] expected = new double[num_classes];
            expected[correct_ans]=1;
            return Utils.add_matrices(neural_output, Utils.multiply(expected, -1));
        }
        private int getMaxIndex(double[] input){
            double max = 0;
            int index = 0;

            for(int i=0;i<input.length;i++){
                if(input[i] >= max){
                    max = input[i];
                    index = i;
                }
            }
            return index;
        }
        public int guess(Image img){
            List<double[][]> img_data = new ArrayList<>();
            img_data.add(Utils.multiply(img.getData(),(1.0/scale_factor)));

            double[] out = layers.get(0).getOutput(img_data);
            return getMaxIndex(out);
        }
        public float test(List<Image> images){
            int correct = 0;
            for(Image i:images){
                int guess = guess(i);
                if(guess == i.getLabel()){
                    correct++;
                }
            }
            return ((float) correct / images.size());
        }
        public void train(List<Image> images){
            for(Image img:images) {
                List<double[][]> img_data = new ArrayList<>();
                img_data.add(Utils.multiply(img.getData(),(1.0/scale_factor)));

                double[] out = layers.get(0).getOutput(img_data);
                double[] dl_out = getErrors(out, img.getLabel());
                // backProp last layers
                layers.get((layers.size()-1)).backProp(dl_out);
            }
        }
    }

    class NetworkBuilder{
        private final int input_rows;
        private final int input_cols;
        private final double scale_factor;
        List<Layer> layers;

        public NetworkBuilder(int input_rows, int input_cols, double scale_factor) {
            this.input_rows = input_rows;
            this.input_cols = input_cols;
            this.scale_factor = scale_factor;
            layers = new ArrayList<>();
        }
        public void addConvolutionLayer(int num_filters, int filter_size, int step_size,
                                        double lr, long SEED) {
            // ConvolutionalLayer(int kernel_size, int stride, int in_channel, int rows, int cols, long SEED, int kernel_num,double lr)
            if(layers.isEmpty()){
                layers.add(new ConvolutionalLayer(filter_size, step_size, 1, input_rows, input_cols, SEED, num_filters, lr));
            } else {
                Layer prev = layers.get(layers.size()-1);
                layers.add(new ConvolutionalLayer(filter_size, step_size, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputRows(), SEED, num_filters, lr));
            }
        }
        public void addMaxPoolLayer(int window, int step_size) {
            // MaxPoolLayer(int stride, int window, int in_channel, int rows, int cols)
            if(layers.isEmpty()){
                layers.add(new MaxPoolLayer(step_size, window, 1, input_rows, input_cols));
            } else {
                Layer prev = layers.get(layers.size()-1);
                layers.add(new MaxPoolLayer(step_size, window, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
            }
        }
        public void addLinearLayer(int out_channel, long SEED, double lr){
            // LinearLayer(int in_channel, int out_channel, long SEED, double learn_rate)
            if(layers.isEmpty()){
                layers.add(new LinearLayer(input_rows*input_cols, out_channel, SEED, lr));
            } else {
                Layer prev = layers.get(layers.size()-1);
                layers.add(new LinearLayer(prev.getOutputElements(), out_channel, SEED, lr));
            }
        }

        public NeuralNet build(){
            return new NeuralNet(layers, scale_factor);
        }
    }

    public static void main(String[] args){
        System.out.println("Loading data");
        List<Image> train_images = new DataReader().readData(train_path);
        List<Image> test_images = new DataReader().readData(test_path);
        System.out.println("Dataset size :");
        System.out.println("Train : "+train_images.size());
        System.out.println("Test : "+test_images.size());

        long SEED = 42;
        MNIST_java c = new MNIST_java();
        NetworkBuilder net = c.new NetworkBuilder(28,28, 256*100);
        net.addConvolutionLayer(8, 5, 1, 0.1,SEED);
        net.addMaxPoolLayer(3, 2);
        net.addLinearLayer(10, SEED, 0.1);

        NeuralNet model = net.build();
        float rate = model.test(test_images);
        System.out.println("Accuracy without training : "+rate);

        int epochs = 10;
        for(int i=0;i<epochs;i++){
            shuffle(train_images);
            model.train(train_images);
            rate = model.test(test_images);
            System.out.println("Accuracy after "+(i+1)+" epoch : "+rate);

        }
    }
}
