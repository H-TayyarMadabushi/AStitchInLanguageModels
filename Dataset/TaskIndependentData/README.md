

# TaskIndependentData

Where possible, please use the training, development and test splits provided so results can remain comparable.

The data format is as follows: 

Each data split (test, dev, ...) has an associated list which represents data associated with a single Multiword Expression (MWE). Each of these has a list, an element of which is an example which uses that MWE. This list contains the following fields: 
* idiom id
* MWE
* Literal Meaning
* Non-literal meaning 1 (or None)
* Non-literal meaning 2 (or None)
* Non-literal meaning 3 (or None)
* Proper Noun (As all MWEs can be used as proper nouns)
* Meta Usage (As all MWEs can be used in this way - please see paper for details)
* 0/1 (1 if this example is not idiomatic, i.e. 1 includes Proper noun and Meta usage)
* The fine grained label associated with this example. 
* The sentence prior to the target sentence containing the MWE. 
* The sentence containing the MWE
* The sentence after the target sentence containing the MWE. 
* The source of these sentences.

Below is part of the English dataset: 

```json
{ "test": 
[
  [
    [
                38,
                "sacred cow",
                "divine cow",
                "above criticism ",
                "None",
                "None",
                "Proper Noun",
                "Meta Usage",
                1,
                "Proper Noun",
                " A C-87 Liberator Express was reconfigured for use as the first dedicated VIP-and-presidential transport aircraft and named Guess Where II, but the Secret Service rejected it because of its safety record.",
                " A C-54 Skymaster was then converted for presidential use; dubbed the Sacred Cow, it carried President Franklin D. Roosevelt to the Yalta Conference in February 1945 and was used for another two years by President Harry S. Truman.",
                "The \"Air Force One\" call sign was created in 1953, after a Lockheed Constellation named Columbine II carrying President Dwight D. Eisenhower entered the same airspace as a commercial airline flight using the same flight number.",
                "https://en.wikipedia.org/wiki/Air_Force_One"
            ],
            
            
            
```
