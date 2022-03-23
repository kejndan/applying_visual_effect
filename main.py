from add_effect import ApplyEffect
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--collection", help="Path to collection", required=True)
    parser.add_argument("-q", "--query", help="Path to query folder", required=True)
    parser.add_argument("-a", "--annotate", help="Path to annotate", default=True)
    parser.add_argument("-e", "--effect", help="Path to effect", default='flare.png')

    # Read arguments from command line
    args = parser.parse_args()

    # calling the main function
    ae = ApplyEffect(path_to_imgs=args.collection,path_to_query_img=args.query,
                     path_to_annotation=args.annotate, path_to_effect=args.effect)
    ae.apply_effect()

