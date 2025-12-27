import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

// Fragmentise tekstove i pravi dataset

public class BuildDataset {

    private static final int FRAG_TOKENS = 400; // velicina fragmenata

    public static void main(String[] args) throws Exception {

        Path rawDir = Paths.get("res");
        Path fragsDir = Paths.get("data_fragments");

        fragmentAllAuthors(rawDir, fragsDir, FRAG_TOKENS);

        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(fragsDir.toFile());
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("author_fragments.arff"));
        saver.writeBatch();

        System.out.println("Dataset saved to author_fragments.arff");
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());
    }


    private static void fragmentAllAuthors(Path rawDir, Path fragsDir, int fragTokens) throws IOException {
        if (!Files.exists(fragsDir)) Files.createDirectories(fragsDir);

        try (Stream<Path> authors = Files.list(rawDir)) {
            for (Path authorDir : authors.filter(Files::isDirectory).collect(Collectors.toList())) {
                String author = authorDir.getFileName().toString();
                Path outAuthor = fragsDir.resolve(author);
                if (!Files.exists(outAuthor)) Files.createDirectories(outAuthor);

                List<Path> books;
                try (Stream<Path> s = Files.list(authorDir)) {
                    books = s.filter(p -> p.toString().toLowerCase().endsWith(".txt"))
                             .collect(Collectors.toList());
                }

                List<String> allTokens = new ArrayList<>();
                for (Path book : books) {
                    String rawText = Files.readString(book, StandardCharsets.UTF_8);
                    String[] toks = rawText.replaceAll("\\s+", " ").trim().split(" ");
                    for (String t : toks) {
                        if (!t.isEmpty()) {
                            allTokens.add(t);
                        }
                    }
                }

                int idx = 0;
                int fragId = 1;
                
                while (idx < allTokens.size()) {
                    int end = Math.min(idx + fragTokens, allTokens.size());
                    
                    if (end - idx < fragTokens / 2) break; // preskok prekratkog poslednjeg fragmenta
                    
                    List<String> frag = allTokens.subList(idx, end);
                    String name = String.format("frag_%04d.txt", fragId++);
                    Path outFile = outAuthor.resolve(name);
                    
                    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(outFile, StandardCharsets.UTF_8))) {
                        pw.println(String.join(" ", frag));
                    }
                    idx = end;
                }
                System.out.printf("Author %-15s -> fragments: %d%n", author, (fragId - 1));
            }
        }
    }
}