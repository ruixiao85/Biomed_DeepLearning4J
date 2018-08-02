package DL4J;

import static DL4J.Core.train;

public class Main {
    public static void main(String[] args){
        UI ui=new UI(
                "D:\\Cel files\\2018-07.13 Adam Brenderia 2X LPS CGS",
                "071318 Cleaned 24H post cgs",
                "2018-07.20 Kyle MMP13 Smoke Flu Zander 2X",
                "Original",
                "Paren",
                12,
                1e-4
        );
        train(ui);
    }

}
