import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Random;


//Ucitava ARFF dataset i radi SMO sa n-gramim recima

public class AuthorID {

    private static final int MIN_TERM_FREQ = 2;
    private static final int NGRAM_MIN = 1;
    private static final int NGRAM_MAX = 3;
    private static final int WORDS_TO_KEEP = 20000;

    
    private static StringToWordVector makeWordNgramFilter(int nMin, int nMax, int minTermFreq, int wordsToKeep) {
        StringToWordVector stwv = new StringToWordVector();
        stwv.setLowerCaseTokens(true);
        stwv.setStopwordsHandler(new Rainbow());
        stwv.setTFTransform(true);
        stwv.setIDFTransform(true);
        stwv.setMinTermFreq(minTermFreq);
        stwv.setWordsToKeep(wordsToKeep);
        stwv.setOutputWordCounts(true);

        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(nMin);
        tokenizer.setNGramMaxSize(nMax);
        tokenizer.setDelimiters("\\W+");
        stwv.setTokenizer(tokenizer);

        return stwv;
    }
    
    
    public static void main(String[] args) throws Exception {

        DataSource src = new DataSource("author_fragments_600.arff");
        Instances raw = src.getDataSet();
        raw.setClassIndex(raw.numAttributes() - 1);

        StringToWordVector stwv = makeWordNgramFilter(NGRAM_MIN, NGRAM_MAX, MIN_TERM_FREQ, WORDS_TO_KEEP);
        stwv.setInputFormat(raw);
        System.out.println("Converting text to n-gram vectors...");
        Instances data = Filter.useFilter(raw, stwv);
        System.out.println("Vectorization done.");

        SMO smo = new SMO();

        System.out.println("Running SMO classifier...");
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(smo, data, 10, new Random(42));
        System.out.println("SMO evaluation done.");
        System.out.println();
        System.out.println();

        System.out.printf("Accuracy: %.2f%%%n", (1.0 - eval.errorRate()) * 100.0);
        System.out.println();
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}