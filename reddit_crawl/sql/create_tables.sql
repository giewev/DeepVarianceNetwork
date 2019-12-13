# ************************************************************
# Sequel Pro SQL dump
# Version 4541
#
# http://www.sequelpro.com/
# https://github.com/sequelpro/sequelpro
#
# Host: 127.0.0.1 (MySQL 5.6.35)
# Database: nlp_final
# Generation Time: 2019-12-13 00:18:48 +0000
# ************************************************************


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;


# Dump of table comments_master
# ------------------------------------------------------------

DROP TABLE IF EXISTS `comments_master`;

CREATE TABLE `comments_master` (
  `comment_id` varchar(100) NOT NULL DEFAULT '',
  `comment_text` text NOT NULL,
  PRIMARY KEY (`comment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



# Dump of table final_result
# ------------------------------------------------------------

DROP TABLE IF EXISTS `final_result`;

CREATE TABLE `final_result` (
  `redditor` varchar(100) CHARACTER SET utf8 NOT NULL DEFAULT '',
  `user_id` varchar(100) CHARACTER SET utf8 NOT NULL DEFAULT '',
  `on_bipolar` int(1) NOT NULL DEFAULT '0',
  `comment_text` text NOT NULL,
  `commented_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `subreddit` varchar(100) CHARACTER SET utf8 NOT NULL DEFAULT '',
  `comment_karma` int(11) NOT NULL,
  `link_karma` int(11) NOT NULL,
  `acct_created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `on_bpd` int(1) NOT NULL,
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



# Dump of table redditor_comments
# ------------------------------------------------------------

DROP TABLE IF EXISTS `redditor_comments`;

CREATE TABLE `redditor_comments` (
  `user_id` varchar(100) NOT NULL DEFAULT '',
  `comment_id` varchar(100) NOT NULL,
  `subreddit` varchar(100) NOT NULL DEFAULT '',
  `commented_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `inserted_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `user_id` (`user_id`,`comment_id`),
  KEY `subreddit` (`subreddit`),
  KEY `commented_at` (`commented_at`),
  KEY `user_id_2` (`user_id`),
  KEY `comment_id` (`comment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



# Dump of table redditors
# ------------------------------------------------------------

DROP TABLE IF EXISTS `redditors`;

CREATE TABLE `redditors` (
  `user_id` varchar(100) NOT NULL DEFAULT '',
  `name` varchar(100) NOT NULL DEFAULT '',
  `comment_karma` int(11) NOT NULL,
  `link_karma` int(11) NOT NULL,
  `source_subreddit` varchar(100) NOT NULL DEFAULT '',
  `acct_created_at` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `inserted_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`),
  KEY `source_subreddit` (`source_subreddit`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



# Dump of table user_class
# ------------------------------------------------------------

DROP TABLE IF EXISTS `user_class`;

CREATE TABLE `user_class` (
  `user_id` varchar(100) CHARACTER SET utf8 NOT NULL DEFAULT '',
  `on_bipolar` int(1) NOT NULL DEFAULT '0',
  `on_bpd` int(1) NOT NULL DEFAULT '0',
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;




/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
