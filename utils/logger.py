# coding=utf-8
import logging


def setup_logging(args):
    """로깅 설정"""
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    return logger
