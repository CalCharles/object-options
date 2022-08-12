class TransformerPair(PairNetwork):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param token_sizes: the layers of the conv network
        """
        args.no_nets = True
        super().__init__(args)

        self.emb = args.embedding_dim
        self.heads = args.heads
        self.depth = args.depth

        conv_args = copy.deepcopy(args)
        conv_args.object_dim = args.pair.object_dim + max(0, self.first_obj_dim * int(not self.drop_first)) + args.pair.object_dim * int(args.pair.difference_first)
        self.conv_dim = self.hs[-1] + max(0, self.post_dim) if args.pair.aggregate_final else args.num_outputs    
        conv_args.output_dim = self.conv_dim
        conv_args.include_last = True #args.pair.aggregate_final
        if args.pair.aggregate_final: conv_args.activation_final = "none" # the final  activation is after the aggregated MLP

        self.conv_args.hs = args.transformer.token_sizes
        self.conv_args.output_dim = self.emb
        self.token_embedding = ConvNetwork(self.conv_args)
        self.layers.append(self.token_embedding)


        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False, dropout=dropout))
            self.layers.append(tblocks[-1])

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, self.output_dim)


    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        print(x.shape)
        return F.log_softmax(x, dim=1)
