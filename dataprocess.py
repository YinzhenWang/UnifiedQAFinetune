from init import *

# Load Scifact dataset
scifact_corpus = load_dataset("scifact", "corpus")
scifact_claims = load_dataset("scifact", "claims")

scifact_train_claim_dict = dict()
scifact_validation_claim_dict = dict()
scifact_corpus_dict = dict()

# Training set
for claim in scifact_claims["train"]:
    scifact_train_claim_dict[claim["id"]] = claim

# Validation set
for claim in scifact_claims["validation"]:
    scifact_validation_claim_dict[claim["id"]] = claim

# Corpus set
for instance in scifact_corpus["train"]:
    scifact_corpus_dict[instance["doc_id"]] = instance

def data_prepare(file_name, dataset_name, flag=False):
    sources = list()
    targets = list()

    scifact_claim_dict = scifact_train_claim_dict if dataset_name == "train" else scifact_validation_claim_dict

    options = "(A) yes (B) no"

    with open(Path(k_data_dir) / file_name, "r") as data_file:
        for line in data_file.readlines():
            cid, question = line.split("\t")
            cid = int(cid)
            question = question.strip()

            gt_claim = scifact_claim_dict[cid]

            label = gt_claim["evidence_label"]

            if label:
                gt_evidence_doc_id = int(gt_claim["evidence_doc_id"])
                gt_corpus = scifact_corpus_dict[gt_evidence_doc_id]

                for index in gt_claim["evidence_sentences"]:
                    gt_evidence_sentence = gt_corpus["abstract"][index].strip()

                    sources += [question + " \\n " + options + " \\n " + gt_evidence_sentence]
                    if flag:
                        targets += ["yes" if label == "SUPPORT" else "no"]
                    else:
                        targets += ["yes" if random.choice([True, False]) else "no"]

                # TODO: random sample negative sentence for the "no answer" option

            else:
                pass

    with open(Path(k_data_dir) / (dataset_name + ".source"), "w") as source_file, open(Path(k_data_dir) / (dataset_name + ".target"), "w") as target_file:
        source_file.writelines("\n".join(sources))
        target_file.writelines("\n".join(targets))

data_prepare("train.rb.qg.tsv", "train")
data_prepare("dev.rb.qg.tsv", "val")