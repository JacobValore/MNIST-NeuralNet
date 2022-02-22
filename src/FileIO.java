import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import javax.imageio.ImageIO;

public class FileIO {
	public static Node[] readImageToNode(String pathName) {
		Node[] nodes = new Node[784];
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(pathName));
		} catch (IOException e) {
			e.printStackTrace();
		}
		int index = 0;
		for(int i = 0; i < 28; i++) {
			for(int j = 0; j < 28; j++) {
				Color c = new Color(img.getRGB(i,j));
				double activation = c.getRed();
				nodes[index] = new Node(0, index, activation/255);
				index++;
			}
		}
		
		return nodes;
	}
	
	public static boolean saveTotalCostToFile(String pathName, ArrayList<Double> costList) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(pathName)));
			boolean first = true;
			for(Double d : costList) {
				String s = d.toString();
				if(first) {
					first = false;
				}
				else { 
					writer.newLine();
				}
				writer.write(s);
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return true;
	}
	
	public static ArrayList<Double> readFile(String pathName) {
		ArrayList<Double> dList = new ArrayList<Double>();
		try {
			Scanner reader = new Scanner(new FileReader(new File(pathName)));
			while(reader.hasNextLine()) {
				String s = reader.nextLine();
				dList.add(Double.valueOf(s));
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dList;
	}
	
	static File searchFile(File file, String search) {
	    if (file.isDirectory()) {
	        File[] arr = file.listFiles();
	        for (File f : arr) {
	            File found = searchFile(f, search);
	            if (found != null)
	                return found;
	        }
	    } else {
	        if (file.getName().equals(search)) {
	            return file;
	        }
	    }
	    return null;
	}
	
	public static String findFile(String dirName, String fileName) {
		File f = new File(dirName);
		String path = searchFile(f,fileName).getAbsolutePath();
		Main.target = Integer.parseInt(path.substring(f.getAbsolutePath().length()+1, f.getAbsolutePath().length()+2));
		return path;
	}

	public static Node[] convertToNode(int[][] pixels) {
		Node[] list = new Node[pixels.length*pixels[0].length];
		int k = 0;
		for(int i = 0; i < pixels.length; i++) {
			for(int j = 0; j < pixels.length; j++) {
				list[k] = new Node(0,k,pixels[i][j]/255.0);
				k++;
			}
		}
		return list;
	}
}
